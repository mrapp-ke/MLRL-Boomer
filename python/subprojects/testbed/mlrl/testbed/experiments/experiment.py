"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import replace
from itertools import chain
from typing import Any, Callable, Generator, Iterable, List, Optional, Set, override

from mlrl.testbed.arguments import PredictionDatasetArguments
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


class ExperimentListener(ABC):
    """
    An abstract base class for all listeners that may be informed about certain event during an experiment.
    """

    def before_start(self, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to be notified just before the experiment starts.

        :param state:   The current state of the experiment
        :return:        An update of the given state
        """
        return state

    def on_start(self, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to be notified when an experiment has been started on a specific
        dataset. May be called multiple times if several datasets are used.

        :param state:   The current state of the experiment
        :return:        An update of the given state
        """
        return state

    def before_training(self, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to be notified before a machine learning model is trained.

        :param state:   The current state of the experiment
        :return:        An update of the given state
        """
        return state

    def after_training(self, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to be notified after a machine learning model has been trained.

        :param state:   The current state of the experiment
        :return:        An update of the given state
        """
        return state

    def after_prediction(self, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to be notified after predictions for a dataset have been obtained
        from a machine learning model. May be called multiple times if predictions are obtained for several
        datasets.

        :param state:   The current state of the experiment
        :return:        An update of the given state
        """
        return state


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
            self.listeners: List[ExperimentListener] = []
            self.input_readers: List[InputReader] = []
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
            self.exit_on_missing_input = False
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

        def add_listeners(self, *listeners: ExperimentListener) -> 'Experiment.Builder':
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
            self.input_readers.extend(input_readers)
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

        def set_exit_on_missing_input(self, exit_on_missing_input: bool) -> 'Experiment.Builder':
            """
            Sets whether the program should exit if an error occurs while reading input data.

            :param exit_on_missing_input:   Tue, if the program should be aborted if an error occurs, False otherwise
            :return:                        The builder itself
            """
            self.exit_on_missing_input = exit_on_missing_input
            return self

        def build(self, args: Namespace) -> 'Experiment':
            """
            Creates and returns a new experiment according to the specified configuration.

            :param args:    The command line arguments specified by the user
            :return:        The experiment that has been created
            """
            exit_on_error = self.exit_on_error

            for output_writer in chain(self.pre_training_output_writers, self.post_training_output_writers,
                                       self.prediction_output_writers):
                output_writer.exit_on_error = exit_on_error

            exit_on_missing_input = self.exit_on_missing_input

            for input_reader in self.input_readers:
                for source in input_reader.sources:
                    source.exit_on_missing_input = exit_on_missing_input

            experiment = self._create_experiment(args, self.initial_state, self.dataset_splitter)
            experiment.listeners.extend(self.listeners)

            def sort(objects: Iterable[Any]) -> List[Any]:
                return sorted(objects, key=lambda obj: type(obj).__name__)

            experiment.input_readers.extend(sort(self.input_readers))
            experiment.before_start_output_writers.extend(sort(self.before_start_output_writers))
            experiment.pre_training_output_writers.extend(sort(self.pre_training_output_writers))
            experiment.post_training_output_writers.extend(sort(self.post_training_output_writers))
            experiment.prediction_output_writers.extend(sort(self.prediction_output_writers))
            return experiment

        def run(self, args: Namespace):
            """
            Creates and runs a new experiment according to the specified configuration.

            :param args: The command line arguments specified by the user
            """
            should_predict = any(bool(output_writer.sinks) for output_writer in self.prediction_output_writers)
            procedure = DefaultProcedure(
                predict_for_training_dataset=should_predict and self.predict_for_training_dataset,
                predict_for_test_dataset=should_predict and self.predict_for_test_dataset,
            )
            procedure.conduct_experiment(self.build(args))

        @abstractmethod
        def _create_experiment(self, args: Namespace, initial_state: ExperimentState,
                               dataset_splitter: DatasetSplitter) -> 'Experiment':
            """
            Must be implemented by subclasses in order to create a new experiment.

            :param args:                The command line arguments specified by the user
            :param initial_state:       The initial state of the experiment
            :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
            :return:                    The experiment that has been created
            """

    class InputReaderListener(ExperimentListener):
        """
        Updates the state of an experiment by invoking the input readers that have been added to an experiment.
        """

        def __init__(self, experiment: 'Experiment', args: Namespace):
            """
            :param experiment:  The experiment
            :param args:        The command line arguments specified by the user
            """
            self.experiment = experiment
            self.args = args

        @override
        def on_start(self, state: ExperimentState) -> ExperimentState:
            original_dataset_type = state.dataset_type

            for input_reader in self.experiment.input_readers:
                input_data = input_reader.input_data
                context = input_data.context
                dataset_types: List[DatasetType] = []

                if context.include_dataset_type:
                    args = self.args
                    predict_for_training_dataset = PredictionDatasetArguments.PREDICT_FOR_TRAINING_DATA.get_value(args)

                    if predict_for_training_dataset:
                        dataset_types.append(DatasetType.TRAINING)

                    predict_for_test_dataset = PredictionDatasetArguments.PREDICT_FOR_TEST_DATA.get_value(args)

                    if predict_for_test_dataset:
                        dataset_types.append(DatasetType.TEST)
                else:
                    dataset_types.append(original_dataset_type)

                for dataset_type in dataset_types:
                    state.dataset_type = dataset_type

                    if not context.include_dataset_type:
                        state = input_reader.read(state)
                    else:
                        if any(source.is_available(state, input_data) for source in input_reader.sources):
                            state = input_reader.read(state)
                        elif len(dataset_types) == 1 and dataset_types[0] == DatasetType.TEST:
                            state.dataset_type = DatasetType.TRAINING
                            state = input_reader.read(state)

            state.dataset_type = original_dataset_type
            return state

    class OutputWriterListener(ExperimentListener):
        """
        Passes the state of an experiment to output writers that have been added to an experiment.
        """

        def __init__(self, experiment: 'Experiment'):
            """
            :param experiment The experiment
            """
            self.experiment = experiment

        @override
        def before_start(self, state: ExperimentState):
            for output_writer in self.experiment.before_start_output_writers:
                output_writer.write(state)

            return state

        @override
        def before_training(self, state: ExperimentState) -> ExperimentState:
            for output_writer in self.experiment.pre_training_output_writers:
                output_writer.write(state)

            return state

        @override
        def after_training(self, state: ExperimentState) -> ExperimentState:
            for output_writer in self.experiment.post_training_output_writers:
                output_writer.write(state)

            return state

        @override
        def after_prediction(self, state: ExperimentState):
            for output_writer in self.experiment.prediction_output_writers:
                output_writer.write(state)

            return state

    class TrainingProcedure(ABC):
        """
        An abstract base class for all classes that allow to fit a learner to a training dataset.
        """

        @abstractmethod
        def train(self, learner: Optional[Any], parameters: ParameterDict, dataset: Dataset) -> TrainingState:
            """
            Fits a learner to a training dataset.

            :param learner:     An existing learner or None, if a new learner must be trained from scratch
            :param parameters:  The algorithmic parameters to be used
            :param dataset:     The training dataset
            :return:            A `TrainingState` that stores the result of the training process
            """

    class PredictionProcedure(ABC):
        """
        An abstract base class for all classes that allow to obtain predictions for given query examples from a
        previously trained learner.
        """

        @abstractmethod
        def predict(self, state: ExperimentState) -> Generator[PredictionState, None, None]:
            """
            Obtains predictions for given query examples from a previously trained learner.

            :param state:   The current state of the experiment
            :return:        The `PredictionState` that stores the result of the prediction process
            """

    def __init__(self, args: Namespace, initial_state: ExperimentState, dataset_splitter: DatasetSplitter,
                 training_procedure: TrainingProcedure, prediction_procedure: PredictionProcedure):
        """
        :param args:                    The command line arguments specified by the user
        :param initial_state:           The initial state of the experiment
        :param dataset_splitter:        The method to be used for splitting the dataset into training and test datasets
        :param training_procedure:      The procedure that allows to fit a learner
        :param prediction_procedure:    The procedure that allows to obtain predictions from a learner
        """
        self.initial_state = initial_state
        self.dataset_splitter = dataset_splitter
        self.training_procedure = training_procedure
        self.prediction_procedure = prediction_procedure
        self.input_readers: List[InputReader] = []
        self.before_start_output_writers: List[OutputWriter] = []
        self.pre_training_output_writers: List[OutputWriter] = []
        self.post_training_output_writers: List[OutputWriter] = []
        self.prediction_output_writers: List[OutputWriter] = []
        self.listeners: List[ExperimentListener] = [
            Experiment.InputReaderListener(self, args),
            Experiment.OutputWriterListener(self),
        ]


class ExperimentalProcedure(ABC):
    """
    An abstract base class for all classes that implement procedures for conducting experiments.
    """

    def conduct_experiment(self, experiment: Experiment) -> ExperimentState:
        """
        Conducts a given experiment.

        :param experiment:  The experiment to be conducted
        :return:            The final state of the experiment
        """
        state = self._before_experiment(experiment, experiment.initial_state)
        state = self._conduct_experiment(experiment, state)
        return self._after_experiment(experiment, state)

    # pylint: disable=unused-argument
    def _before_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to perform an operation before an experiment starts.

        :param experiment:  The experiment
        :param state:       The current state of the experiment
        :return:            An updated state
        """
        return state

    @abstractmethod
    def _conduct_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        """
        Must be implemented by subclasses in order to conduct an experiment.

        :param experiment:  The experiment
        :param state:       The current state of the experiment
        :return:            An updated state
        """

    # pylint: disable=unused-argument
    def _after_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        """
        May be overridden by subclasses in order to perform an operation after an experiment has been completed.

        :param experiment:  The experiment
        :param state:       The current state of the experiment
        :return:            An updated state
        """
        return state


class DefaultProcedure(ExperimentalProcedure):
    """
    Implements the default procedure for conducting experiments.
    """

    EXTRA_START_TIME = 'start_time'

    @staticmethod
    def __predict(experiment: Experiment, state: ExperimentState):
        prediction_results = experiment.prediction_procedure.predict(state)

        for prediction_result in prediction_results:
            new_state = replace(state, prediction_result=prediction_result)

            for listener in experiment.listeners:
                new_state = listener.after_prediction(new_state)

    def __init__(self, predict_for_training_dataset: bool, predict_for_test_dataset: bool):
        """
        :param predict_for_training_dataset:    True, if predictions should be obtained for the training dataset, False
                                                otherwise
        :param predict_for_test_dataset:        True, if predictions should be obtained for the test dataset, False
                                                otherwise
        """
        self.predict_for_training_dataset = predict_for_training_dataset
        self.predict_for_test_dataset = predict_for_test_dataset

    @override
    def _before_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        problem_domain = state.problem_domain
        log.info('Starting experiment using the %s algorithm "%s"...', problem_domain.problem_name,
                 problem_domain.learner_name)

        for listener in experiment.listeners:
            state = listener.before_start(state)

        state.extras[self.EXTRA_START_TIME] = Timer.start()
        return state

    @override
    def _conduct_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        listeners = experiment.listeners

        for split in experiment.dataset_splitter.split(state):
            training_state = split.get_state(DatasetType.TRAINING)

            if training_state:
                for listener in listeners:
                    training_state = listener.on_start(training_state)

                for listener in listeners:
                    training_state = listener.before_training(training_state)

                # Train model...
                training_result = experiment.training_procedure.train(
                    learner=training_state.training_result.learner if training_state.training_result else None,
                    parameters=training_state.parameters,
                    dataset=training_state.dataset)
                training_state = replace(training_state, training_result=training_result)
                test_state = split.get_state(DatasetType.TEST)

                # Obtain and evaluate predictions for training data, if necessary...
                if self.predict_for_training_dataset or (self.predict_for_test_dataset and not test_state):
                    self.__predict(experiment, training_state)

                # Obtain and evaluate predictions for test data, if necessary...
                if test_state and self.predict_for_test_dataset:
                    self.__predict(experiment, replace(test_state, training_result=training_result))

                for listener in listeners:
                    training_state = listener.after_training(training_state)

        return state

    @override
    def _after_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
        start_time = state.extras.get(self.EXTRA_START_TIME)

        if start_time:
            run_time = Timer.stop(start_time)
            log.info('Successfully finished experiment after %s', run_time)
        else:
            log.info('Successfully finished experiment')

        return state
