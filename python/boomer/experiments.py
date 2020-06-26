#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC

from sklearn.base import clone

from boomer.common.learners import Learner, NominalAttributeLearner
from boomer.data import MetaData, AttributeType
from boomer.evaluation import Evaluation
from boomer.parameters import ParameterInput
from boomer.printing import ModelPrinter
from boomer.training import CrossValidation, DataSet


class Experiment(CrossValidation, ABC):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, base_learner: Learner, data_set: DataSet, num_folds: int = 1, current_fold: int = -1,
                 train_evaluation: Evaluation = None, test_evaluation: Evaluation = None,
                 parameter_input: ParameterInput = None, model_printer: ModelPrinter = None):
        """
        :param base_learner:        The classifier or ranker to be trained
        :param train_evaluation:    The evaluation to be used for evaluating the predictions for the training data or
                                    None, if the predictions should not be evaluated
        :param test_evaluation:     The evaluation to be used for evaluating the predictions for the test data or None,
                                    if the predictions should not be evaluated
        :param parameter_input:     The input that should be used to read the parameter settings
        :param model_printer:       The printer that should be used to print textual representations of models
        """
        super().__init__(data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.parameter_input = parameter_input
        self.model_printer = model_printer

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')
        super().run()

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(current_fold)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        # Train classifier
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)

        current_learner.fit(train_x, train_y)
        learner_name = current_learner.get_name()

        # Obtain and evaluate predictions for training data, if necessary
        evaluation = self.train_evaluation

        if evaluation is not None:
            predictions = current_learner.predict(train_x)
            evaluation.evaluate('train_' + learner_name, predictions, train_y, first_fold=first_fold,
                                current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)

        # Obtain and evaluate predictions for test data, if necessary
        evaluation = self.test_evaluation

        if evaluation is not None:
            predictions = current_learner.predict(test_x)
            evaluation.evaluate('test_' + learner_name, predictions, test_y, first_fold=first_fold,
                                current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)

        # Print model, if necessary
        model_printer = self.model_printer

        if model_printer is not None:
            model_printer.print(learner_name, meta_data, current_learner, current_fold=current_fold,
                                num_folds=num_folds)
