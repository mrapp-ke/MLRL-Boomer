#!/usr/bin/python

import argparse
import logging as log

import numpy as np
import sklearn.metrics as metrics
from sklearn.base import clone
from sklearn.model_selection import KFold
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.model import DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.algorithm.rule_learners import MLRuleLearner
from boomer.experiments import CrossValidation
from boomer.learners import Randomized
from main import configure_argument_parser, create_learner


class GridSearch(MLClassifierBase, Randomized):
    PARAMETERS = [0.25]  # [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.5]

    def __init__(self, base_learner: MLRuleLearner, num_folds: int, min_rules: int):
        super().__init__()
        self.base_learner = base_learner
        self.num_folds = num_folds
        self.min_rules = min_rules
        self.require_dense = [True, True]  # We need a dense representation of the training data

    def fit(self, x, y):
        base_learner = self.base_learner
        num_folds = self.num_folds
        random_state = self.random_state
        min_rules = self.min_rules
        avg_performances = None
        num_parameters = len(GridSearch.PARAMETERS)
        i = 0
        k_fold = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

        for train, test in k_fold.split(x, y):
            log.info('Nested fold %s / %s:', (i + 1), num_folds)

            # Create training set for current fold
            train_x = np.asfortranarray(self._ensure_input_format(x[train]), dtype=DTYPE_FLOAT32)
            train_y = self._ensure_input_format(y[train])

            # Create test set for current fold
            test_x = np.asfortranarray(self._ensure_input_format(x[test]), dtype=DTYPE_FLOAT32)
            test_y = self._ensure_input_format(y[test])

            for g in range(num_parameters):
                log.info('Testing parameter setting %s / %s:', (g + 1), num_parameters)

                # Train classifier
                current_learner = clone(base_learner)
                current_learner.set_params(**{'shrinkage': GridSearch.PARAMETERS[g]})
                current_learner.random_state = random_state
                current_learner.fold = i
                current_learner.fit(train_x, train_y)

                # Evaluate classifier
                theory = current_learner.theory_
                num_rules = len(theory)
                predictions = np.asfortranarray(np.zeros((test_x.shape[0], test_y.shape[1])), dtype=DTYPE_FLOAT64)

                if avg_performances is None:
                    avg_performances = np.zeros((len(GridSearch.PARAMETERS), num_rules - min_rules + 1), dtype=float)

                for j in range(num_rules):
                    rule = theory.pop(0)
                    rule.predict(test_x, predictions)

                    if j + 1 >= min_rules:
                        performance = metrics.hamming_loss(test_y, np.where(predictions > 0, 1, 0))
                        avg_performances[g, j - min_rules + 1] += performance

            i += 1

        avg_performances = np.divide(avg_performances, num_folds)
        min_row, min_col = np.divmod(avg_performances.argmin(), avg_performances.shape[1])

        best_shrinkage = GridSearch.PARAMETERS[min_row.item()]
        best_num_rules = min_col + min_rules
        best_performance = avg_performances[min_row, min_col]
        print('best_shrinkage=' + str(best_shrinkage) + ', best_num_rules=' + str(
            best_num_rules) + ', best_performance=' + str(best_performance))

    def predict(self, x):
        pass


class ParameterTuning(CrossValidation):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, current_fold: int, base_learner: MLRuleLearner,
                 num_nested_folds: int, min_rules: int):
        super().__init__(data_dir, data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.num_nested_folds = num_nested_folds
        self.min_rules = min_rules

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, first_fold: int, current_fold: int, last_fold: int,
                            num_folds: int):
        grid_search = GridSearch(self.base_learner, num_folds=self.num_nested_folds, min_rules=self.min_rules)
        grid_search.random_state = self.random_state
        grid_search.fit(train_x, train_y)


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='Tunes the parameters of a BOOMER model using nested cross validation')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--nested-folds', type=int, default=5,
                        help='Total number of folds to be used by nested cross validation')
    parser.add_argument('--min-rules', type=int, default=100, help='The minimum number of rules')
    configure_argument_parser(parser)
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    learner = create_learner(args)
    parameter_tuning = ParameterTuning(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                       current_fold=args.current_fold, num_nested_folds=args.nested_folds,
                                       min_rules=args.min_rules, base_learner=learner)
    parameter_tuning.random_state = args.random_state
    parameter_tuning.run()
