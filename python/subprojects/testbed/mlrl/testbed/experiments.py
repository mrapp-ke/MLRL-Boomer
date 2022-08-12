"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from timeit import default_timer as timer
from typing import Optional

from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter
from mlrl.testbed.data_splitting import DataSplitter, DataSplit, DataType
from mlrl.testbed.evaluation import Evaluation
from mlrl.testbed.model_characteristics import ModelPrinter, ModelCharacteristicsPrinter
from mlrl.testbed.parameters import ParameterInput, ParameterPrinter
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsPrinter
from mlrl.testbed.predictions import PredictionPrinter
from sklearn.base import BaseEstimator, RegressorMixin, clone


class PredictionType(Enum):
    """
    Contains all possible types of predictions that may be obtained from a learner.
    """
    LABELS = 'labels'
    SCORES = 'scores'
    PROBABILITIES = 'probabilities'


class Experiment(DataSplitter.Callback):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    class ExecutionHook(ABC):
        """
        An abstract base class for all operations that may be executed before or after an experiment.
        """

        @abstractmethod
        def execute(self):
            """
            Must be overridden by subclasses in order to execute the operation.
            """
            pass

    def __init__(self,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 data_splitter: DataSplitter,
                 pre_execution_hook: Optional[ExecutionHook] = None,
                 prediction_type: PredictionType = PredictionType.LABELS,
                 train_evaluation: Optional[Evaluation] = None,
                 test_evaluation: Optional[Evaluation] = None,
                 train_prediction_printer: Optional[PredictionPrinter] = None,
                 test_prediction_printer: Optional[PredictionPrinter] = None,
                 train_prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter] = None,
                 test_prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter] = None,
                 parameter_input: Optional[ParameterInput] = None, parameter_printer: Optional[ParameterPrinter] = None,
                 model_printer: Optional[ModelPrinter] = None,
                 model_characteristics_printer: Optional[ModelCharacteristicsPrinter] = None,
                 data_characteristics_printer: Optional[DataCharacteristicsPrinter] = None,
                 persistence: Optional[ModelPersistence] = None):
        """
        :param base_learner:                                The classifier or ranker to be trained
        :param learner_name:                                The name of the classifier or ranker
        :param data_splitter:                               The method to be used for splitting the available data into
                                                            training and test sets
        :param pre_execution_hook:                          An operation that should be executed before the experiment
        :param prediction_type:                             The type of the predictions to be obtained from the
                                                            classifier or ranker
        :param train_evaluation:                            The evaluation to be used for evaluating the predictions for
                                                            the training data or None, if the predictions should not be
                                                            evaluated
        :param test_evaluation:                             The evaluation to be used for evaluating the predictions for
                                                            the test data or None, if the predictions should not be
                                                            evaluated
        :param train_prediction_printer:                    The printer that should be used to print the predictions for
                                                            the training data or None, if the predictions should not be
                                                            printed
        :param test_prediction_printer                      The printer that should be used to print the predictions for
                                                            the test data or None, if the predictions should not be
                                                            printed
        :param train_prediction_characteristics_printer:    The printer that should be used to print the characteristics
                                                            of binary predictions for the training data or None, if the
                                                            characteristics should not be printed
        :param test_prediction_characteristics_printer:     The printer that should be used to print the characteristics
                                                            of binary predictions for the test data or None, if the
                                                            characteristics should not be printed
        :param parameter_input:                             The input that should be used to read the parameter settings
        :param parameter_printer:                           The printer that should be used to print parameter settings
        :param model_printer:                               The printer that should be used to print textual
                                                            representations of models or None, if no textual
                                                            representations should be printed
        :param model_characteristics_printer:               The printer that should be used to print the characteristics
                                                            of models or None, if the characteristics should not be
                                                            printed
        :param data_characteristics_printer:                The printer that should be used to print the characteristics
                                                            of the training data or None, if the characteristics should
                                                            not be printed
        :param persistence:                                 The `ModelPersistence` that should be used for loading and
                                                            saving models
        """
        self.base_learner = base_learner
        self.learner_name = learner_name
        self.data_splitter = data_splitter
        self.pre_execution_hook = pre_execution_hook
        self.prediction_type = prediction_type
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.train_prediction_printer = train_prediction_printer
        self.test_prediction_printer = test_prediction_printer
        self.train_prediction_characteristics_printer = train_prediction_characteristics_printer
        self.test_prediction_characteristics_printer = test_prediction_characteristics_printer
        self.parameter_input = parameter_input
        self.parameter_printer = parameter_printer
        self.model_printer = model_printer
        self.model_characteristics_printer = model_characteristics_printer
        self.data_characteristics_printer = data_characteristics_printer
        self.persistence = persistence

    def run(self):
        log.info('Starting experiment...')

        # Run pre-execution hook, if necessary...
        if self.pre_execution_hook is not None:
            self.pre_execution_hook.execute()

        self.data_splitter.run(self)

    def train_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, train_x, train_y, test_x, test_y):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary...
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(data_split)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        # Print parameter setting, if necessary...
        parameter_printer = self.parameter_printer

        if parameter_printer is not None:
            parameter_printer.print(data_split, current_learner)

        # Print data characteristics, if necessary...
        data_characteristics_printer = self.data_characteristics_printer

        if data_characteristics_printer is not None:
            data_characteristics_printer.print(meta_data, data_split, train_x, train_y)

        # Set the indices of nominal attributes, if supported...
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)

        # Load model from disc, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(data_split)

        if isinstance(loaded_learner, type(current_learner)):
            current_params = current_learner.get_params()
            self.__check_for_parameter_changes(expected_params=current_params,
                                               actual_params=loaded_learner.get_params())
            loaded_learner.set_params(**current_params)
            current_learner = loaded_learner
            train_time = 0
        else:
            log.info('Fitting model to %s training examples...', train_x.shape[0])
            train_time = self.__train(current_learner, train_x, train_y)
            log.info('Successfully fit model in %s seconds', train_time)

            # Save model to disk...
            self.__save_model(current_learner, data_split)

        # Obtain and evaluate predictions for training data, if necessary...
        evaluation = self.train_evaluation
        prediction_printer = self.train_prediction_printer
        prediction_characteristics_printer = None if self.prediction_type != PredictionType.LABELS else \
            self.train_prediction_characteristics_printer

        if evaluation is not None or prediction_printer is not None or prediction_characteristics_printer is not None:
            log.info('Predicting for %s training examples...', train_x.shape[0])
            predictions, predict_time = self.__predict(current_learner, train_x)

            if predictions is not None:
                data_type = DataType.TRAINING

                if evaluation is not None:
                    evaluation.evaluate(meta_data, data_split, data_type, predictions, train_y,
                                        train_time=train_time, predict_time=predict_time)

                if prediction_printer is not None:
                    prediction_printer.print(meta_data, data_split, data_type, predictions, train_y)

                if prediction_characteristics_printer is not None:
                    prediction_characteristics_printer.print(data_split, data_type, predictions)

        # Obtain and evaluate predictions for test data, if necessary...
        evaluation = self.test_evaluation
        prediction_printer = self.test_prediction_printer
        prediction_characteristics_printer = None if self.prediction_type != PredictionType.LABELS else \
            self.test_prediction_characteristics_printer

        if evaluation is not None or prediction_printer is not None or prediction_characteristics_printer is not None:
            log.info('Predicting for %s test examples...', test_x.shape[0])
            predictions, predict_time = self.__predict(current_learner, test_x)

            if predictions is not None:
                data_type = DataType.TEST

                if evaluation is not None:
                    evaluation.evaluate(meta_data, data_split, data_type, predictions, test_y,
                                        train_time=train_time, predict_time=predict_time)

                if prediction_printer is not None:
                    prediction_printer.print(meta_data, data_split, data_type, predictions, test_y)

                if prediction_characteristics_printer is not None:
                    prediction_characteristics_printer.print(data_split, data_type, predictions)

        # Print model characteristics, if necessary...
        model_characteristics_printer = self.model_characteristics_printer

        if model_characteristics_printer is not None:
            try:
                model_characteristics_printer.print(data_split, current_learner)
            except ValueError:
                log.error('The learner does not support to obtain model characteristics')

        # Print model, if necessary...
        model_printer = self.model_printer

        if model_printer is not None:
            try:
                model_printer.print(meta_data, data_split, current_learner)
            except ValueError:
                log.error('The learner does not support to create a textual representation of the model')

    @staticmethod
    def __train(learner, x, y):
        """
        Fits a learner to training data.

        :param learner: The learner
        :param x:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                        the feature values of the training examples
        :param y:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                        labels of the training examples according to the ground truth
        :return:        The time needed for training
        """
        start_time = timer()
        learner.fit(x, y)
        end_time = timer()
        return end_time - start_time

    def __predict(self, learner, x):
        """
        Obtains predictions from a learner.

        :param learner: The learner
        :param x:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                        the feature values of the query examples
        :return:        The predictions, as well as the time needed to obtain them
        """
        start_time = timer()
        prediction_type = self.prediction_type

        if prediction_type == PredictionType.SCORES:
            try:
                if isinstance(learner, Learner):
                    predictions = learner.predict(x, predict_scores=True)
                elif isinstance(learner, RegressorMixin):
                    predictions = learner.predict(x)
                else:
                    raise RuntimeError()
            except RuntimeError:
                log.error('Prediction of regression scores not supported')
                predictions = None
        elif prediction_type == PredictionType.PROBABILITIES:
            try:
                predictions = learner.predict_proba(x)
            except RuntimeError:
                log.error('Prediction of probabilities not supported')
                predictions = None
        else:
            predictions = learner.predict(x)

        end_time = timer()
        predict_time = end_time - start_time

        if predictions is not None:
            log.info('Successfully predicted in %s seconds', predict_time)

        return predictions, predict_time

    def __load_model(self, data_split: DataSplit):
        """
        Loads the model from disk, if available.

        :param data_split:  Information about the split of the available data, the model corresponds to
        :return:            The loaded model
        """
        persistence = self.persistence

        if persistence is not None:
            return persistence.load_model(self.learner_name, data_split)

        return None

    def __save_model(self, model, data_split: DataSplit):
        """
        Saves a model to disk.

        :param model:       The model to be saved
        :param data_split:  Information about the split of the available data, the model corresponds to
        """
        persistence = self.persistence

        if persistence is not None:
            persistence.save_model(model, self.learner_name, data_split)

    @staticmethod
    def __check_for_parameter_changes(expected_params, actual_params):
        changes = []

        for key, expected_value in expected_params.items():
            actual_value = actual_params[key]

            if actual_value != expected_value:
                changes.append((key, expected_value, actual_value))

        if len(changes) > 0:
            log.warning(
                'The loaded model\'s values for the following parameters differ from the expected configuration: %s',
                reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b[0] + '" is "' + str(
                    b[2]) + '" instead of "' + str(b[1] + '"'), changes, ''))
