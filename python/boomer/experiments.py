#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC
from typing import List

from sklearn.base import clone

from boomer.evaluation import Evaluation
from boomer.learners import MLLearner, NominalAttributeLearner
from boomer.parameters import ParameterInput
from boomer.printing import ModelPrinter
from boomer.training import CrossValidation, DataSet


class Experiment(CrossValidation, ABC):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, base_learner: MLLearner, evaluation: Evaluation, data_set: DataSet, num_folds: int = 1,
                 current_fold: int = -1, parameter_input: ParameterInput = None, model_printer: ModelPrinter = None):
        """
        :param base_learner:    The classifier or ranker to be trained
        :param parameter_input: The input that should be used to read the parameter settings
        :param model_printer:   The printer that should be used to print textual representations of models
        """
        super().__init__(data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.evaluation = evaluation
        self.parameter_input = parameter_input
        self.model_printer = model_printer

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')
        super().run()

    def _train_and_evaluate(self, nominal_attribute_indices: List[int], train_indices, train_x, train_y, test_indices,
                            test_x, test_y, first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(current_fold)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        # Train classifier
        current_learner.random_state = self.random_state
        current_learner.fold = current_fold

        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = nominal_attribute_indices

        current_learner.fit(train_x, train_y)
        learner_name = current_learner.get_name()

        # Obtain and evaluate predictions for test data
        predictions = current_learner.predict(test_x)
        self.evaluation.evaluate(learner_name, predictions, test_y, first_fold=first_fold, current_fold=current_fold,
                                 last_fold=last_fold, num_folds=num_folds)

        # Print model, if necessary
        model_printer = self.model_printer

        if model_printer is not None:
            model_printer.print(learner_name, current_learner, current_fold=current_fold, num_folds=num_folds)
