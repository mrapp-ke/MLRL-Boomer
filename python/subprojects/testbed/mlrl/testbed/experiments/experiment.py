"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import replace
from functools import reduce
from itertools import chain
from typing import Any, Callable, Generator, Iterable, List, Optional, Set, override

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.meta_data.writer import MetaDataWriter
from mlrl.testbed.experiments.output.model.writer import ModelWriter
from mlrl.testbed.experiments.output.parameters.writer import ParameterWriter
from mlrl.testbed.experiments.output.sinks import FileSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState, ParameterDict, PredictionState, TrainingState
from mlrl.testbed.experiments.timer import Timer


class Experiment(ABC):
    """
    An abstract base class for all experiments that train and evaluate a machine learning model.
    """

    class Builder(ABC):
        """
        An abstract base class for all classes that allow to configure and create instances of an experiment.
        """

        Factory = Callable[[Namespace], 'Experiment.Builder']

        def __init__(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter):
            """
            :param initial_state:       The initial state of the experiment
            :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
            """
            super().__init__()
            self.initial_state = initial_state
            self.dataset_splitter = dataset_splitter
            self.listeners: List[Experiment.Listener] = []
            self.input_readers: Set[InputReader] = set()
            self.before_start_output_writers: Set[OutputWriter] = set()
            self.pre_training_output_writers: Set[OutputWriter] = set()
            self.post_training_output_writers: Set[OutputWriter] = set()
            self.prediction_output_writers: Set[OutputWriter] = set()
            self.model_writer = ModelWriter()
            self.meta_data_writer = MetaDataWriter()
            self.parameter_writer = ParameterWriter()
            self.predict_for_training_dataset = False
            self.predict_for_test_dataset = True
            self.exit_on_error = True
            self.add_before_start_output_writers(self.meta_data_writer)
            self.add_pre_training_output_writers(self.parameter_writer)
            self.add_post_training_output_writers(self.model_writer)

        @property
        def output_writers(self) -> chain[OutputWriter]:
            """
            A generator that provides access to all output writers that have been added to the builder.
            """
            return chain(self.before_start_output_writers, self.pre_training_output_writers,
                         self.post_training_output_writers, self.prediction_output_writers)

        @property
        def has_output_file_writers(self) -> bool:
            """
            True, if any output writers that write to output files have been added to the builder, False otherwise.
            """
            return any(any(isinstance(sink, FileSink) for sink in writer.sinks) for writer in self.output_writers)

        def add_listeners(self, *listeners: 'Experiment.Listener') -> 'Experiment.Builder':
            """
            Adds one or several listeners that should be informed about certain events during the experiment.

            :param listeners:   The listeners that should be added
            :return:            The builder itself
            """
            self.listeners.extend(listeners)
            return self

        def add_input_readers(self, *input_readers: InputReader) -> 'Experiment.Builder':
            """
            Adds one or several input readers that should be invoked when the experiment is started.

            :param input_readers:   The input readers to be added
            :return:                The builder itself
            """
            self.input_readers.update(input_readers)
            return self

        def add_before_start_output_writers(self, *output_writers: OutputWriter) -> 'Experiment.Builder':
            """
            Adds one or several output writers that should be invoked before an experiment is started.

            :param output_writers:  The output writers to be added
            :return:                The builder itself
            """
            self.before_start_output_writers.update(output_writers)
            return self

        def add_pre_training_output_writers(self, *output_writers: OutputWriter) -> 'Experiment.Builder':
            """
            Adds one or several output writers that should be invoked before a machine learning model is trained.

            :param output_writers:  The output writers to be added
            :return:                The builder itself
            """
            self.pre_training_output_writers.update(output_writers)
            return self

        def add_post_training_output_writers(self, *output_writers: OutputWriter) -> 'Experiment.Builder':
            """
            Adds one or several output writers that should be invoked after a machine learning model has been trained.

            :param output_writers:  The output writers to be added
            :return:                The builder itself
            """
            self.post_training_output_writers.update(output_writers)
            return self

        def add_prediction_output_writers(self, *output_writers: OutputWriter) -> 'Experiment.Builder':
            """
            Adds one or several output writers that should be invoked after predictions have been obtained from a
            machine learning model.

            :param output_writers:  The output writers to be added
            :return:                The builder itself
            """
            self.prediction_output_writers.update(output_writers)
            return self

        def set_predict_for_training_dataset(self, predict_for_training_dataset: bool) -> 'Experiment.Builder':
            """
            Sets whether predictions should be obtained for the training dataset or not.

            :param predict_for_training_dataset:    True, if predictions should be obtained for the training dataset,
                                                    False otherwise
            :return:                                The builder itself
            """
            self.predict_for_training_dataset = predict_for_training_dataset
            return self

        def set_predict_for_test_dataset(self, predict_for_test_dataset: bool) -> 'Experiment.Builder':
            """
            Sets whether predictions should be obtained for the test dataset, if available, or not.

            :param predict_for_test_dataset:    True, if predictions should be obtained for the test dataset, if
                                                available, False otherwise
            :return:                            The builder itself
            """
            self.predict_for_test_dataset = predict_for_test_dataset
            return self

        def set_exit_on_error(self, exit_on_error: bool) -> 'Experiment.Builder':
            """
            Sets whether the program should exit if an error occurs while writing experimental results.

            :param exit_on_error:   True, if the program should be aborted if an error occurs, False otherwise
            :return:                The builder itself
            """
            self.exit_on_error = exit_on_error
            return self

        def build(self) -> 'Experiment':
            """
            Creates and returns a new experiment according to the specified configuration.

            :return: The experiment that has been created
            """
            exit_on_error = self.exit_on_error

            for output_writer in chain(self.pre_training_output_writers, self.post_training_output_writers,
                                       self.prediction_output_writers):
                output_writer.exit_on_error = exit_on_error

            experiment = self._create_experiment(self.initial_state, self.dataset_splitter)
            experiment.listeners.extend(self.listeners)

            def sort(objects: Iterable[Any]) -> List[Any]:
                return sorted(objects, key=lambda obj: type(obj).__name__)

            experiment.input_readers.extend(sort(self.input_readers))
            experiment.before_start_output_writers.extend(sort(self.before_start_output_writers))
            experiment.pre_training_output_writers.extend(sort(self.pre_training_output_writers))
            experiment.post_training_output_writers.extend(sort(self.post_training_output_writers))
            experiment.prediction_output_writers.extend(sort(self.prediction_output_writers))
            return experiment

        def run(self):
            """
            Creates and runs a new experiment according to the specified configuration.
            """
            should_predict = any(bool(output_writer.sinks) for output_writer in self.prediction_output_writers)
            self.build().run(predict_for_training_dataset=should_predict and self.predict_for_training_dataset,
                             predict_for_test_dataset=should_predict and self.predict_for_test_dataset)

        @abstractmethod
        def _create_experiment(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter) -> 'Experiment':
            """
            Must be implemented by subclasses in order to create a new experiment.

            :param initial_state:       The initial state of the experiment
            :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
            :return:                    The experiment that has been created
            """

    class Listener(ABC):
        """
        An abstract base class for all listeners that may be informed about certain event during an experiment.
        """

        # pylint: disable=unused-argument
        def before_start(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            """
            May be overridden by subclasses in order to be notified just before the experiment starts.

            :param experiment:  The experiment
            :param state:       The current state of the experiment
            :return:            An update of the given state
            """
            return state

        # pylint: disable=unused-argument
        def on_start(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            """
            May be overridden by subclasses in order to be notified when an experiment has been started on a specific
            dataset. May be called multiple times if several datasets are used.

            :param experiment:  The experiment
            :param state:       The current state of the experiment
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

        @override
        def on_start(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            return reduce(lambda current_state, input_reader: input_reader.read(current_state),
                          experiment.input_readers, state)

    class OutputWriterListener(Listener):
        """
        Passes the state of an experiment to output writers that have been added to an experiment.
        """

        @override
        def before_start(self, experiment: 'Experiment', state: ExperimentState):
            for output_writer in experiment.before_start_output_writers:
                output_writer.write(state)

            return state

        @override
        def before_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            for output_writer in experiment.pre_training_output_writers:
                output_writer.write(state)

            return state

        @override
        def after_training(self, experiment: 'Experiment', state: ExperimentState) -> ExperimentState:
            for output_writer in experiment.post_training_output_writers:
                output_writer.write(state)

            return state

        @override
        def after_prediction(self, experiment: 'Experiment', state: ExperimentState):
            for output_writer in experiment.prediction_output_writers:
                output_writer.write(state)

    def __predict(self, state: ExperimentState):
        prediction_results = self._predict(state)

        for prediction_result in prediction_results:
            new_state = replace(state, prediction_result=prediction_result)

            for listener in self.listeners:
                listener.after_prediction(self, new_state)

    def __init__(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter):
        """
        :param initial_state:       The initial state of the experiment
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        """
        self.initial_state = initial_state
        self.dataset_splitter = dataset_splitter
        self.input_readers: List[InputReader] = []
        self.before_start_output_writers: List[OutputWriter] = []
        self.pre_training_output_writers: List[OutputWriter] = []
        self.post_training_output_writers: List[OutputWriter] = []
        self.prediction_output_writers: List[OutputWriter] = []
        self.listeners = [
            Experiment.InputReaderListener(),
            Experiment.OutputWriterListener(),
        ]

    def run(self, predict_for_training_dataset: bool, predict_for_test_dataset: bool):
        """
        Runs the experiment.

        :param predict_for_training_dataset:    True, if predictions should be obtained for the training dataset, False
                                                otherwise
        :param predict_for_test_dataset:        True, if predictions should be obtained for the test dataset, if
                                                available, False otherwise
        """
        initial_state = self.initial_state
        problem_domain = initial_state.problem_domain
        log.info('Starting experiment using the %s algorithm "%s"...', problem_domain.problem_name,
                 problem_domain.learner_name)

        for listener in self.listeners:
            listener.before_start(self, initial_state)

        start_time = Timer.start()

        for split in self.dataset_splitter.split(initial_state):
            training_state = split.get_state(DatasetType.TRAINING)

            if training_state:
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
        log.info('Successfully finished experiment after %s', run_time)

    @abstractmethod
    def _train(self, learner: Optional[Any], parameters: ParameterDict, dataset: Dataset) -> TrainingState:
        """
        Must be implemented by subclasses in order to fit a learner to a training dataset.

        :param learner: An existing learner or None, if a new learner must be trained from scratch
        :param dataset: The training dataset
        :return:        A `TrainingState` that stores the result of the training process
        """

    @abstractmethod
    def _predict(self, state: ExperimentState) -> Generator[PredictionState, None, None]:
        """
        Must be implemented by subclasses in order to obtain predictions for given query examples from a previously
        trained learner.

        :param state:   The current state of the experiment
        :return:        The `PredictionState` that stores the result of the prediction process
        """
