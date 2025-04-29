"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator, clone

from mlrl.common.mixins import NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed.experiments.dataset import AttributeType, Dataset, DatasetType
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction import Predictor
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState, TrainingState
from mlrl.testbed.experiments.timer import Timer


class Experiment:
    """
    An experiment that trains and evaluates a machine learning model on a specific data set using cross validation or
    separate training and test sets.
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

    def __init__(self,
                 problem_type: ProblemType,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 dataset_splitter: DatasetSplitter,
                 input_readers: List[InputReader],
                 pre_training_output_writers: List[OutputWriter],
                 post_training_output_writers: List[OutputWriter],
                 prediction_output_writers: List[OutputWriter],
                 pre_execution_hook: Optional[ExecutionHook] = None,
                 train_predictor: Optional[Predictor] = None,
                 test_predictor: Optional[Predictor] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param problem_type:                    The type of the machine learning problem
        :param base_learner:                    The machine learning algorithm to be used
        :param learner_name:                    The name of the machine learning algorithm
        :param dataset_splitter:                The method to be used for splitting the dataset into training and test
                                                datasets
        :param input_readers:                   A list that contains all input readers to be invoked
        :param pre_training_output_writers:     A list that contains all output writers to be invoked before training
        :param post_training_output_writers:    A list that contains all output writers to be invoked after training
        :param prediction_output_writers:       A list that contains all output writers to be invoked each time
                                                predictions are obtained from a model
        :param pre_execution_hook:              An operation that should be executed before the experiment
        :param train_predictor:                 The `Predictor` to be used for obtaining predictions for the training
                                                data or None, if no such predictions should be obtained
        :param test_predictor:                  The `Predictor` to be used for obtaining predictions for the test data
                                                or None, if no such predictions should be obtained
        :param fit_kwargs:                      Optional keyword arguments to be passed to the learner when fitting a
                                                model
        :param predict_kwargs:                  Optional keyword arguments to be passed to the learner when obtaining
                                                predictions from a model
        """
        self.problem_type = problem_type
        self.base_learner = base_learner
        self.learner_name = learner_name
        self.dataset_splitter = dataset_splitter
        self.input_readers = input_readers
        self.pre_training_output_writers = pre_training_output_writers
        self.prediction_output_writers = prediction_output_writers
        self.post_training_output_writers = post_training_output_writers
        self.pre_execution_hook = pre_execution_hook
        self.train_predictor = train_predictor
        self.test_predictor = test_predictor
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

    # pylint: disable=too-many-branches
    def run(self):
        """
        Runs the experiment.
        """
        log.info('Starting experiment using the %s algorithm "%s"...', self.problem_type.value, self.learner_name)

        # Run pre-execution hook, if necessary...
        pre_execution_hook = self.pre_execution_hook

        if pre_execution_hook:
            pre_execution_hook.execute()

        start_time = Timer.start()

        for split in self.dataset_splitter.split(problem_type=self.problem_type):
            training_state = split.get_state(DatasetType.TRAINING)
            training_dataset = training_state.dataset

            # Read input data...
            for input_reader in self.input_readers:
                input_reader.read(training_state)

            # Apply parameter setting, if necessary...
            learner = clone(self.base_learner)
            parameters = training_state.parameters

            if parameters:
                learner.set_params(**parameters)
                log.info('Successfully applied parameter setting: %s', parameters)
            else:
                parameters = learner.get_params()

            # Write output data before model is trained...
            for output_writer in self.pre_training_output_writers:
                output_writer.write(training_state)

            # Set the indices of ordinal features, if supported...
            fit_kwargs = self.fit_kwargs if self.fit_kwargs else {}

            if isinstance(learner, OrdinalFeatureSupportMixin):
                fit_kwargs[
                    OrdinalFeatureSupportMixin.KWARG_ORDINAL_FEATURE_INDICES] = training_dataset.get_feature_indices(
                        AttributeType.ORDINAL)

            # Set the indices of nominal features, if supported...
            if isinstance(learner, NominalFeatureSupportMixin):
                fit_kwargs[
                    NominalFeatureSupportMixin.KWARG_NOMINAL_FEATURE_INDICES] = training_dataset.get_feature_indices(
                        AttributeType.NOMINAL)

            # Load model from disk, if possible, otherwise train a new model...
            loaded_learner = training_state.training_result.learner if training_state.training_result else None

            if isinstance(loaded_learner, type(learner)):
                self.__check_for_parameter_changes(expected_params=parameters,
                                                   actual_params=loaded_learner.get_params())
                loaded_learner.set_params(**parameters)
                learner = loaded_learner
                training_duration = training_state.training_result.training_duration
            else:
                log.info('Fitting model to %s training examples...', training_dataset.num_examples)
                training_duration = self.__train(learner, training_dataset, **fit_kwargs)
                log.info('Successfully fit model in %s', training_duration)

            training_result = TrainingState(learner=learner, training_duration=training_duration)
            training_state = replace(training_state, training_result=training_result)
            test_state = replace(split.get_state(DatasetType.TEST), training_result=training_result)

            # Obtain and evaluate predictions for training data, if necessary...
            test_dataset = test_state.dataset
            train_predictor = self.train_predictor

            if train_predictor and test_dataset.type != DatasetType.TRAINING:
                predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
                self.__predict_and_evaluate(training_state, train_predictor, **predict_kwargs)

            # Obtain and evaluate predictions for test data, if necessary...
            test_predictor = self.test_predictor

            if test_predictor:
                predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
                self.__predict_and_evaluate(test_state, test_predictor, **predict_kwargs)

            # Write output data after model was trained...
            for output_writer in self.post_training_output_writers:
                output_writer.write(training_state)

        run_time = Timer.stop(start_time)
        log.info('Successfully finished after %s', run_time)

    def __predict_and_evaluate(self, state: ExperimentState, predictor: Predictor, **kwargs):
        """
        Obtains predictions for given query examples from a previously trained model.

        :param state:       The state that stores the model
        :param predictor:   The `Predictor` to be used for obtaining the predictions
        :param kwargs:      Optional keyword arguments to be passed to the model when obtaining predictions
        """
        try:
            for prediction_state in predictor.obtain_predictions(state, **kwargs):
                new_state = replace(state, prediction_result=prediction_state)

                for output_writer in self.prediction_output_writers:
                    output_writer.write(new_state)
        except ValueError as error:
            dataset = state.dataset

            if dataset.has_sparse_features:
                dense_dataset = replace(state, dataset=dataset.enforce_dense_features())
                Experiment.__predict_and_evaluate(dense_dataset, predictor, **kwargs)

            raise error

    @staticmethod
    def __train(learner, dataset: Dataset, **kwargs) -> Timer.Duration:
        """
        Fits a learner to training data.

        :param learner: The learner
        :param dataset: The training dataset
        :param kwargs:  Optional keyword arguments to be passed to the learner when fitting model
        :return:        The time needed for training
        """
        try:
            start_time = Timer.start()
            learner.fit(dataset.x, dataset.y, **kwargs)
            return Timer.stop(start_time)
        except ValueError as error:
            if dataset.has_sparse_features:
                return Experiment.__train(learner, dataset.enforce_dense_features(), **kwargs)
            if dataset.has_sparse_outputs:
                return Experiment.__train(learner, dataset.enforce_dense_outputs(), **kwargs)
            raise error

    @staticmethod
    def __check_for_parameter_changes(expected_params, actual_params):
        changes = []

        for key, expected_value in expected_params.items():
            expected_value = str(expected_value)
            actual_value = str(actual_params[key])

            if actual_value != expected_value:
                changes.append((key, expected_value, actual_value))

        if changes:
            log.warning(
                'The loaded model\'s values for the following parameters differ from the expected configuration: %s',
                reduce(
                    lambda aggr, change: aggr +
                    (', '
                     if aggr else '') + '"' + change[0] + '" is "' + change[2] + '" instead of "' + change[1] + '"',
                    changes, ''))
