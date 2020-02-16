#!/usr/bin/python

import argparse
import logging as log
from typing import List

import numpy as np
from sklearn.base import clone
from skmultilearn.base import MLClassifierBase

from args import float_list
from boomer.algorithm.model import DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.algorithm.rule_learners import Boomer
from boomer.measures import squared_error_loss, logistic_loss
from boomer.parameters import NestedCrossValidation, ParameterTuning, ParameterLogOutput, ParameterCsvOutput
from main_boomer import configure_argument_parser, create_learner


class ShrinkageNumRulesParameterSearch(NestedCrossValidation, MLClassifierBase):
    """
    Allows to tune the parameters "shrinkage" and "num_rules" using (nested) cross validation.
    """

    def __init__(self, num_folds: int, min_rules: int, parameters: List[float], base_learner: Boomer):
        super().__init__(num_folds)
        self.min_rules = min_rules
        self.parameters = parameters
        self.base_learner = base_learner
        self.scores = None
        self.best_params = None
        self.best_score = None
        self.require_dense = [True, True]  # We need a dense representation of the training data

        if base_learner.loss == 'macro-squared-error-loss':
            self.target_measure = squared_error_loss
        elif base_learner.loss == 'example-based-logistic-loss':
            self.target_measure = logistic_loss
        else:
            raise ValueError('Unknown loss function used')

    def _test_parameters(self, train_x, train_y, test_x, test_y, current_outer_fold: int, num_outer_folds: int,
                         current_nested_fold: int, num_nested_folds: int):
        random_state = self.random_state
        base_learner = self.base_learner
        num_rules = base_learner.num_rules
        min_rules = self.min_rules
        parameters = self.parameters
        num_parameters = len(parameters)

        if current_nested_fold == 0:
            scores = np.zeros((num_parameters, num_rules - min_rules + 1), dtype=float)
            self.scores = scores
        else:
            scores = self.scores

        train_x = np.asfortranarray(self._ensure_input_format(train_x), dtype=DTYPE_FLOAT32)
        train_y = self._ensure_input_format(train_y)
        test_x = np.asfortranarray(self._ensure_input_format(test_x), dtype=DTYPE_FLOAT32)
        test_y = self._ensure_input_format(test_y)

        for g in range(num_parameters):
            log.info('Testing parameter setting %s / %s (Fold %s / %s):', (g + 1), num_parameters,
                     (current_nested_fold + 1), num_nested_folds)

            # Train classifier
            current_learner = clone(base_learner)
            current_learner.set_params(**{'shrinkage': parameters[g]})
            current_learner.random_state = random_state
            current_learner.fold = (current_outer_fold * num_nested_folds) + current_nested_fold
            current_learner.fit(train_x, train_y)

            # Evaluate classifier
            theory = current_learner.theory_
            num_rules = len(theory)
            predictions = np.asfortranarray(np.zeros((test_x.shape[0], test_y.shape[1])), dtype=DTYPE_FLOAT64)

            for j in range(num_rules):
                rule = theory.pop(0)
                rule.predict(test_x, predictions)

                if j + 1 >= min_rules:
                    score = logistic_loss(test_y, predictions)
                    scores[g, j - min_rules + 1] += score

        if current_nested_fold == num_nested_folds - 1:
            scores = np.divide(scores, self.num_nested_folds)
            row, col = np.divmod(scores.argmin(), scores.shape[1])
            best_shrinkage = self.parameters[row.item()]
            best_num_rules = col + self.min_rules
            self.best_params = {'shrinkage': best_shrinkage, 'num_rules': best_num_rules}
            self.best_score = scores[row, col]

    def get_params(self):
        return self.best_params

    def get_score(self):
        return self.best_score

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tunes the hyper-parameters of BOOMER using nested cross validation')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--nested-folds', type=int, default=5,
                        help='Total number of folds to be used by nested cross validation')
    parser.add_argument('--min-rules', type=int, default=100,
                        help='The minimum value of the parameter \'num_rules\' to be tested')
    parser.add_argument('--shrinkage-parameters', type=float_list, default='0.25',
                        help='The parameters \'shrinkage\' to be tested as a comma-separated list')
    configure_argument_parser(parser)
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    parameter_outputs = [ParameterLogOutput()]

    if args.output_dir is not None:
        parameter_outputs.append(ParameterCsvOutput(output_dir=args.output_dir,
                                                    clear_dir=args.current_fold == -1))

    learner = create_learner(args)
    parameter_search = ShrinkageNumRulesParameterSearch(args.nested_folds, args.min_rules, args.shrinkage_parameters,
                                                        learner)
    parameter_tuning = ParameterTuning(args.data_dir, args.dataset, args.folds, args.current_fold, parameter_search,
                                       *parameter_outputs)
    parameter_tuning.random_state = args.random_state
    parameter_tuning.run()
