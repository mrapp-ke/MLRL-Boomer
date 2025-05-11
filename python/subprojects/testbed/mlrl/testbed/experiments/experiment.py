"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from typing import Any, Generator, Optional

from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.problem_domain import ProblemDomain
from mlrl.testbed.experiments.state import ExperimentState, ParameterDict, PredictionState, TrainingState
from mlrl.testbed.experiments.timer import Timer


class Experiment(ABC):
    """
    An abstract base class for all experiments that train and evaluate a machine learning model.
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

        def after_prediction(self, experiment: 'Experiment', state: ExperimentState):
            """
            May be overridden by subclasses in order to be notified after predictions for a dataset have been obtained
            from a machine learning model. May be called multiple times if predictions are obtained for several
            datasets.

            :param experiment:  The experiment
            :param state:       The current state of the experiment
            """

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

        def after_prediction(self, experiment: 'Experiment', state: ExperimentState):
            for output_writer in experiment.prediction_output_writers:
                output_writer.write(state)

    def __predict(self, state: ExperimentState):
        for prediction_result in self._predict(learner=state.training_result.learner,
                                               dataset=state.dataset,
                                               dataset_type=state.dataset_type):
            new_state = replace(state, prediction_result=prediction_result)

            for listener in self.listeners:
                listener.after_prediction(self, new_state)

    def __init__(self, problem_domain: ProblemDomain, dataset_splitter: DatasetSplitter):
        """
        :param problem_domain:      The problem domain, the experiment is concerned with
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        """
        self.problem_domain = problem_domain
        self.dataset_splitter = dataset_splitter
        self.input_readers = []
        self.pre_training_output_writers = []
        self.post_training_output_writers = []
        self.prediction_output_writers = []
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

    def add_prediction_output_writers(self, *output_writers: OutputWriter) -> 'Experiment':
        """
        Adds one or several output writers that should be invoked after predictions have been obtained from a machine
        learning model.

        :param output_writers:  The output writers to be added
        :return:                The experiment itself
        """
        for output_writer in output_writers:
            self.prediction_output_writers.append(output_writer)
        return self

    # pylint: disable=too-many-branches
    def run(self, predict_for_training_dataset: bool = False, predict_for_test_dataset: bool = True):
        """
        Runs the experiment.

        :param predict_for_training_dataset:    True, if predictions should be obtained for the training dataset, False
                                                otherwise
        :param predict_for_test_dataset:        True, if predictions should be obtained for the test dataset, if
                                                available, False otherwise
        """
        problem_domain = self.problem_domain
        log.info('Starting experiment using the %s algorithm "%s"...', problem_domain.problem_type.value,
                 problem_domain.learner_name)

        for listener in self.listeners:
            listener.before_start(self)

        start_time = Timer.start()

        for split in self.dataset_splitter.split(problem_domain):
            training_state = split.get_state(DatasetType.TRAINING)

            for listener in self.listeners:
                training_state = listener.on_start(self, training_state)

            for listener in self.listeners:
                training_state = listener.before_training(self, training_state)

            # Train model...
            training_result = self._train(
                learner=training_state.training_result.learner if training_state.training_result else None,
                parameters=training_state.parameters,
                dataset=training_state.dataset)
            training_state = replace(training_state, training_result=training_result)
            test_state = split.get_state(DatasetType.TEST)

            # Obtain and evaluate predictions for training data, if necessary...
            if predict_for_training_dataset or (predict_for_test_dataset and not test_state):
                self.__predict(training_state)

            # Obtain and evaluate predictions for test data, if necessary...
            if test_state and predict_for_test_dataset:
                self.__predict(replace(test_state, training_result=training_result))

            for listener in self.listeners:
                listener.after_training(self, training_state)

        run_time = Timer.stop(start_time)
        log.info('Successfully finished after %s', run_time)

    @abstractmethod
    def _train(self, learner: Optional[Any], parameters: ParameterDict, dataset: Dataset) -> TrainingState:
        """
        Must be implemented by subclasses in order to fit a learner to a training dataset.

        :param learner: An existing learner or None, if a new learner must be trained from scratch
        :param dataset: The training dataset
        :return:        A `TrainingState` that stores the result of the training process
        """

    @abstractmethod
    def _predict(self, learner: Any, dataset: Dataset,
                 dataset_type: DatasetType) -> Generator[PredictionState, None, None]:
        """
        Must be implemented by subclasses in order to obtain predictions for given query examples from a previously
        trained learner.

        :param learner:         The learner
        :param dataset:         The dataset that contains the query examples
        :param dataset_type:    The type of the dataset
        :return:                The `PredictionState` that stores the result of the prediction process
        """
