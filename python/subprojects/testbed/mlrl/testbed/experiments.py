"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC
from timeit import default_timer as timer
from typing import Optional

from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter
from mlrl.testbed.evaluation import Evaluation
from mlrl.testbed.model_characteristics import ModelPrinter, ModelCharacteristicsPrinter
from mlrl.testbed.parameters import ParameterInput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsPrinter
from mlrl.testbed.predictions import PredictionPrinter
from mlrl.testbed.training import CrossValidation, DataSet
from sklearn.base import clone


class Experiment(CrossValidation, ABC):
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
                 base_learner: Learner,
                 data_set: DataSet,
                 num_folds: int = 1,
                 current_fold: int = -1,
                 pre_execution_hook: Optional[ExecutionHook] = None,
                 predict_probabilities: bool = False,
                 train_evaluation: Optional[Evaluation] = None,
                 test_evaluation: Optional[Evaluation] = None,
                 train_prediction_printer: Optional[PredictionPrinter] = None,
                 test_prediction_printer: Optional[PredictionPrinter] = None,
                 train_prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter] = None,
                 test_prediction_characteristics_printer: Optional[PredictionCharacteristicsPrinter] = None,
                 parameter_input: Optional[ParameterInput] = None,
                 model_printer: Optional[ModelPrinter] = None,
                 model_characteristics_printer: Optional[ModelCharacteristicsPrinter] = None,
                 data_characteristics_printer: Optional[DataCharacteristicsPrinter] = None,
                 persistence: Optional[ModelPersistence] = None):
        """
        :param base_learner:                                The classifier or ranker to be trained
        :param pre_execution_hook:                          An operation that should be executed before the experiment
        :param predict_probabilities:                       True, if probabilities should be predicted rather than
                                                            binary labels, False otherwise
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
        super().__init__(data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.pre_execution_hook = pre_execution_hook
        self.predict_probabilities = predict_probabilities
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.train_prediction_printer = train_prediction_printer
        self.test_prediction_printer = test_prediction_printer
        self.train_prediction_characteristics_printer = train_prediction_characteristics_printer
        self.test_prediction_characteristics_printer = test_prediction_characteristics_printer
        self.parameter_input = parameter_input
        self.model_printer = model_printer
        self.model_characteristics_printer = model_characteristics_printer
        self.data_characteristics_printer = data_characteristics_printer
        self.persistence = persistence

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')

        # Run pre-execution hook, if necessary...
        if self.pre_execution_hook is not None:
            self.pre_execution_hook.execute()

        super().run()

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary...
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(current_fold)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        learner_name = current_learner.get_name()

        # Print data characteristics, if necessary...
        data_characteristics_printer = self.data_characteristics_printer

        if data_characteristics_printer is not None:
            data_characteristics_printer.print(learner_name, train_x, train_y, meta_data, current_fold=current_fold,
                                               num_folds=num_folds)

        # Set the indices of nominal attributes, if supported...
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)

        # Load model from disc, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(model_name=learner_name, current_fold=current_fold, num_folds=num_folds)

        if isinstance(loaded_learner, Learner):
            current_learner = loaded_learner
        else:
            log.info('Fitting model to %s training examples...', train_x.shape[0])
            current_learner.fit(train_x, train_y)
            log.info('Successfully fit model in %s seconds', current_learner.train_time_)

            # Save model to disk...
            self.__save_model(current_learner, current_fold=current_fold, num_folds=num_folds)

        # Obtain and evaluate predictions for training data, if necessary...
        evaluation = self.train_evaluation
        prediction_printer = self.train_prediction_printer
        prediction_characteristics_printer = None if self.predict_probabilities else \
            self.train_prediction_characteristics_printer

        if evaluation is not None or prediction_printer is not None or prediction_characteristics_printer is not None:
            log.info('Predicting for %s training examples...', train_x.shape[0])
            predictions, predict_time = self.__predict(current_learner, train_x)

            if predictions is not None:
                experiment_name = 'train_' + learner_name
                if evaluation is not None:
                    evaluation.evaluate(experiment_name, meta_data, predictions, train_y, first_fold=first_fold,
                                        current_fold=current_fold, last_fold=last_fold, num_folds=num_folds,
                                        train_time=current_learner.train_time_, predict_time=predict_time)

                if prediction_printer is not None:
                    prediction_printer.print(experiment_name, meta_data, predictions, train_y,
                                             current_fold=current_fold, num_folds=num_folds)

                if prediction_characteristics_printer is not None:
                    prediction_characteristics_printer.print(experiment_name, predictions, current_fold=current_fold,
                                                             num_folds=num_folds)

        # Obtain and evaluate predictions for test data, if necessary...
        evaluation = self.test_evaluation
        prediction_printer = self.test_prediction_printer
        prediction_characteristics_printer = None if self.predict_probabilities else \
            self.test_prediction_characteristics_printer

        if evaluation is not None or prediction_printer is not None or prediction_characteristics_printer is not None:
            log.info('Predicting for %s test examples...', test_x.shape[0])
            predictions, predict_time = self.__predict(current_learner, test_x)

            if predictions is not None:
                experiment_name = 'test_' + learner_name

                if evaluation is not None:
                    evaluation.evaluate(experiment_name, meta_data, predictions, test_y, first_fold=first_fold,
                                        current_fold=current_fold, last_fold=last_fold, num_folds=num_folds,
                                        train_time=current_learner.train_time_, predict_time=predict_time)

                if prediction_printer is not None:
                    prediction_printer.print(experiment_name, meta_data, predictions, test_y, current_fold=current_fold,
                                             num_folds=num_folds)

                if prediction_characteristics_printer is not None:
                    prediction_characteristics_printer.print(experiment_name, predictions, current_fold=current_fold,
                                                             num_folds=num_folds)

        # Print model characteristics, if necessary...
        model_characteristics_printer = self.model_characteristics_printer

        if model_characteristics_printer is not None:
            model_characteristics_printer.print(learner_name, current_learner, current_fold=current_fold,
                                                num_folds=num_folds)

        # Print model, if necessary...
        model_printer = self.model_printer

        if model_printer is not None:
            model_printer.print(learner_name, meta_data, current_learner, current_fold=current_fold,
                                num_folds=num_folds)

    def __predict(self, learner, x):
        """
        Obtains predictions from a learner.

        :param learner: The learner
        :param x:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                        the feature values of the query examples
        :return:        The predictions, as well as the time needed
        """
        start_time = timer()

        if self.predict_probabilities:
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

    def __load_model(self, model_name: str, current_fold: int, num_folds: int):
        """
        Loads the model from disk, if available.

        :param model_name:      The name of the model to be loaded
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        :return: The loaded model
        """
        persistence = self.persistence

        if persistence is not None:
            return persistence.load_model(model_name=model_name, fold=(current_fold if num_folds > 1 else None))

        return None

    def __save_model(self, model: Learner, current_fold: int, num_folds: int):
        """
        Saves a model to disk.

        :param model:           The model to be saved
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        persistence = self.persistence

        if persistence is not None:
            persistence.save_model(model, model_name=model.get_name(), fold=(current_fold if num_folds > 1 else None))
