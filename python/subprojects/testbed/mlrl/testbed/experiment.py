"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from abc import ABC
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

    class Listener(ABC):
        """
        An abstract base class for all listeners that may be informed about certain event during an experiment.
        """

        def before_start(self, experiment: 'Experiment'):
            """
            May be overridden by subclasses in order to be notified just before the experiment starts.

            :param experiment:  The experiment
            """

        # pylint: disable=unused-argument
        def on_start(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            """
            May be overridden by subclasses in order to be notified when an experiment has been started on a specific
            dataset. May be called multiple times if several datasets are used.

            :param experiment:  The experiment
            :param state:       The initial state of the experiment
            :return:            An update of the given state
            """
            return state

        # pylint: disable=unused-argument
        def before_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            """
            May be overridden by subclasses in order to be notified before a machine learning model is trained.

            :param experiment:  The experiment
            :param state:       The current state of the experiment
            :return:            An update of the given state
            """
            return state

        # pylint: disable=unused-argument
        def after_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            """
            May be overridden by subclasses in order to be notified after a machine learning model has been trained.

            :param experiment:  The experiment
            :param state:       The current state of the experiment
            :return:            An update of the given state
            """
            return state

    class InputReaderListener(Listener):
        """
        Updates the state of an experiment by invoking the input readers that have been added to an experiment.
        """

        def on_start(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            return reduce(lambda current_state, input_reader: input_reader.read(current_state),
                          experiment.input_readers, state)

    class OutputWriterListener(Listener):
        """
        Passes the state of an experiment to output writers that have been added to an experiment.
        """

        def before_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            for output_writer in experiment.pre_training_output_writers:
                output_writer.write(state)

            return state

        def after_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            for output_writer in experiment.post_training_output_writers:
                output_writer.write(state)

            return state

    def __init__(self,
                 problem_type: ProblemType,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 dataset_splitter: DatasetSplitter,
                 prediction_output_writers: List[OutputWriter],
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
        :param prediction_output_writers:       A list that contains all output writers to be invoked each time
                                                predictions are obtained from a model
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
        self.prediction_output_writers = prediction_output_writers
        self.train_predictor = train_predictor
        self.test_predictor = test_predictor
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.input_readers = []
        self.pre_training_output_writers = []
        self.post_training_output_writers = []
        self.listeners = [
            Experiment.InputReaderListener(),
            Experiment.OutputWriterListener(),
        ]

    def add_listeners(self, *listeners: Listener) -> 'Experiment':
        """
        Adds one or several listeners that should be informed about certain events during the experiment.

        :param listeners:   The listeners that should be added
        :return:            The experiment itself
        """
        for listener in listeners:
            self.listeners.append(listener)
        return self

    def add_input_readers(self, *input_readers: InputReader) -> 'Experiment':
        """
        Adds one or several input readers that should be invoked when an experiment is started.

        :param input_readers:   The input readers to be added
        :return:                The experiment itself
        """
        for input_reader in input_readers:
            self.input_readers.append(input_reader)
        return self

    def add_pre_training_output_writers(self, *output_writers: OutputWriter) -> 'Experiment':
        """
        Adds one or several output writers that should be invoked before a machine learning model is trained.

        :param output_writers:  The output writers to be added
        :return:                The experiment itself
        """
        for output_writer in output_writers:
            self.pre_training_output_writers.append(output_writer)
        return self

    def add_post_training_output_writers(self, *output_writers: OutputWriter) -> 'Experiment':
        """
        Adds one or several output writers that should be invoked after a machine learning model has been trained.

        :param output_writers:  The output writers to be added
        :return:                The experiment itself
        """
        for output_writer in output_writers:
            self.post_training_output_writers.append(output_writer)
        return self

    # pylint: disable=too-many-branches
    def run(self):
        """
        Runs the experiment.
        """
        log.info('Starting experiment using the %s algorithm "%s"...', self.problem_type.value, self.learner_name)

        for listener in self.listeners:
            listener.before_start(self)

        start_time = Timer.start()

        for split in self.dataset_splitter.split(problem_type=self.problem_type):
            training_state = split.get_state(DatasetType.TRAINING)

            for listener in self.listeners:
                training_state = listener.on_start(self, training_state)

            learner = self._create_learner(training_state)

            for listener in self.listeners:
                training_state = listener.before_training(self, training_state)

            # Load model from disk, if possible, otherwise train a new model...
            loaded_learner = training_state.training_result.learner if training_state.training_result else None

            if isinstance(loaded_learner, type(learner)):
                self.__check_for_parameter_changes(expected_params=training_state.parameters,
                                                   actual_params=loaded_learner.get_params())
                training_result = TrainingState(learner=loaded_learner)
            else:
                training_dataset = training_state.dataset
                log.info('Fitting model to %s training examples...', training_dataset.num_examples)
                training_duration = self.__train(learner, training_dataset)
                training_result = TrainingState(learner=learner, training_duration=training_duration)
                log.info('Successfully fit model in %s', training_duration)

            training_state = replace(training_state, training_result=training_result)
            test_state = replace(split.get_state(DatasetType.TEST), training_result=training_result)

            # Obtain and evaluate predictions for training data, if necessary...
            test_dataset = test_state.dataset
            train_predictor = self.train_predictor

            if train_predictor and test_dataset.type != DatasetType.TRAINING:
                self.__predict_and_evaluate(training_state, train_predictor)

            # Obtain and evaluate predictions for test data, if necessary...
            test_predictor = self.test_predictor

            if test_predictor:
                self.__predict_and_evaluate(test_state, test_predictor)

            for listener in self.listeners:
                listener.after_training(self, training_state)

        run_time = Timer.stop(start_time)
        log.info('Successfully finished after %s', run_time)

    def _create_learner(self, state: ExperimentState) -> Any:
        """
        Must be implemented by subclasses in order to create the learner to be used in the experiment.

        :param state:   The state that should be used to store the learner
        :return:        The learner that has been created
        """
        learner = clone(self.base_learner)
        parameters = state.parameters

        if parameters:
            learner.set_params(**parameters)
            log.info('Successfully applied parameter setting: %s', parameters)

        return learner

    def __predict_and_evaluate(self, state: ExperimentState, predictor: Predictor):
        """
        Obtains predictions for given query examples from a previously trained model.

        :param state:       The state that stores the model
        :param predictor:   The `Predictor` to be used for obtaining the predictions
        """
        predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}

        try:
            for prediction_state in predictor.obtain_predictions(state, **predict_kwargs):
                new_state = replace(state, prediction_result=prediction_state)

                for output_writer in self.prediction_output_writers:
                    output_writer.write(new_state)
        except ValueError as error:
            dataset = state.dataset

            if dataset.has_sparse_features:
                dense_dataset = replace(state, dataset=dataset.enforce_dense_features())
                Experiment.__predict_and_evaluate(dense_dataset, predictor, **predict_kwargs)

            raise error

    def __train(self, learner: BaseEstimator, dataset: Dataset) -> Timer.Duration:
        """
        Fits a learner to training data.

        :param learner: The learner
        :param dataset: The training dataset
        :return:        The time needed for training
        """
        # Set the indices of ordinal features, if supported...
        fit_kwargs = self.fit_kwargs if self.fit_kwargs else {}

        if isinstance(learner, OrdinalFeatureSupportMixin):
            fit_kwargs[OrdinalFeatureSupportMixin.KWARG_ORDINAL_FEATURE_INDICES] = dataset.get_feature_indices(
                AttributeType.ORDINAL)

        # Set the indices of nominal features, if supported...
        if isinstance(learner, NominalFeatureSupportMixin):
            fit_kwargs[NominalFeatureSupportMixin.KWARG_NOMINAL_FEATURE_INDICES] = dataset.get_feature_indices(
                AttributeType.NOMINAL)

        try:
            start_time = Timer.start()
            learner.fit(dataset.x, dataset.y, **fit_kwargs)
            return Timer.stop(start_time)
        except ValueError as error:
            if dataset.has_sparse_features:
                return Experiment.__train(learner, dataset.enforce_dense_features(), **fit_kwargs)
            if dataset.has_sparse_outputs:
                return Experiment.__train(learner, dataset.enforce_dense_outputs(), **fit_kwargs)
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
