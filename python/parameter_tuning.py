#!/usr/bin/python

import argparse
import logging as log
from typing import List

import numpy as np
import sklearn.metrics as metrics
from sklearn.base import clone
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.model import DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.algorithm.rule_learners import Boomer
from boomer.parameters import NestedCrossValidation, ParameterTuning, ParameterLogOutput, ParameterCsvOutput
from main import configure_argument_parser, create_learner


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

    def _test_parameters(self, train_x, train_y, test_x, test_y, current_fold: int, num_folds: int):
        random_state = self.random_state
        base_learner = self.base_learner
        num_rules = base_learner.num_rules
        min_rules = self.min_rules
        parameters = self.parameters
        num_parameters = len(parameters)

        if current_fold == 0:
            scores = np.zeros((num_parameters, num_rules - min_rules + 1), dtype=float)
            self.scores = scores
        else:
            scores = self.scores

        train_x = np.asfortranarray(self._ensure_input_format(train_x), dtype=DTYPE_FLOAT32)
        train_y = self._ensure_input_format(train_y)
        test_x = np.asfortranarray(self._ensure_input_format(test_x), dtype=DTYPE_FLOAT32)
        test_y = self._ensure_input_format(test_y)

        for g in range(num_parameters):
            log.info('Testing parameter setting %s / %s:', (g + 1), num_parameters)

            # Train classifier
            current_learner = clone(base_learner)
            current_learner.set_params(**{'shrinkage': parameters[g]})
            current_learner.random_state = random_state
            current_learner.fold = current_fold
            current_learner.fit(train_x, train_y)

            # Evaluate classifier
            theory = current_learner.theory_
            num_rules = len(theory)
            predictions = np.asfortranarray(np.zeros((test_x.shape[0], test_y.shape[1])), dtype=DTYPE_FLOAT64)

            for j in range(num_rules):
                rule = theory.pop(0)
                rule.predict(test_x, predictions)

                if j + 1 >= min_rules:
                    score = metrics.hamming_loss(test_y, np.where(predictions > 0, 1, 0))
                    scores[g, j - min_rules + 1] += score

        if current_fold == num_folds - 1:
            scores = np.divide(scores, self.num_folds)
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


def __float_list(s):
    return [float(x.strip()) for x in s.split(',')]


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='Tunes the parameters of a BOOMER model using nested cross validation')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--nested-folds', type=int, default=5,
                        help='Total number of folds to be used by nested cross validation')
    parser.add_argument('--min-rules', type=int, default=100,
                        help='The minimum value of the parameter \'num_rules\' to be tested')
    parser.add_argument('--shrinkage-parameters', type=__float_list, default='0.25',
                        help='The parameters \'shrinkage\' to be tested as a comma-separated list')
    configure_argument_parser(parser)
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    learner = create_learner(args)
    parameter_search = ShrinkageNumRulesParameterSearch(args.nested_folds, args.min_rules, args.shrinkage_parameters,
                                                        learner)
    parameter_tuning = ParameterTuning(args.data_dir, args.dataset, args.folds, args.current_fold, parameter_search,
                                       ParameterLogOutput(), ParameterCsvOutput(output_dir=args.output_dir,
                                                                                clear_dir=args.current_fold == -1))
    parameter_tuning.random_state = args.random_state
    parameter_tuning.run()
