"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from typing import Any, Callable, Dict, Generator, List, Optional

from sklearn.base import BaseEstimator, clone

from mlrl.common.mixins import NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed.experiments.dataset import AttributeType, Dataset, DatasetType
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction import Predictor
from mlrl.testbed.experiments.problem_type import ProblemType
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

    def __predict(self, state: ExperimentState):
        for prediction_result in self._predict(learner=state.training_result.learner, dataset=state.dataset):
            new_state = replace(state, prediction_result=prediction_result)

            for output_writer in self.prediction_output_writers:
                output_writer.write(new_state)

    def __init__(self, problem_type: ProblemType, learner_name: str, dataset_splitter: DatasetSplitter,
                 prediction_output_writers: List[OutputWriter]):
        """
        :param problem_type:                    The type of the machine learning problem
        :param learner_name:                    The name of the machine learning algorithm
        :param dataset_splitter:                The method to be used for splitting the dataset into training and test
                                                datasets
        :param prediction_output_writers:       A list that contains all output writers to be invoked each time
                                                predictions are obtained from a model
        """
        self.problem_type = problem_type
        self.learner_name = learner_name
        self.dataset_splitter = dataset_splitter
        self.prediction_output_writers = prediction_output_writers
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
    def run(self, predict_for_training_dataset: bool = False, predict_for_test_dataset: bool = True):
        """
        Runs the experiment.

        :param predict_for_training_dataset:    True, if predictions should be obtained for the training dataset, False
                                                otherwise
        :param predict_for_test_dataset:        True, if predictions should be obtained for the test dataset, if
                                                available, False otherwise
        """
        log.info('Starting experiment using the %s algorithm "%s"...', self.problem_type.value, self.learner_name)

        for listener in self.listeners:
            listener.before_start(self)

        start_time = Timer.start()

        for split in self.dataset_splitter.split(problem_type=self.problem_type):
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
    def _predict(self, learner: Any, dataset: Dataset) -> Generator[PredictionState]:
        """
        Must be implemented by subclasses in order to obtain predictions for given query examples from a previously
        trained learner.

        :param learner: The learner
        :param dataset: The dataset that contains the query examples
        :return:        The `PredictionState` that stores the result of the prediction process
        """


class SkLearnExperiment(Experiment):
    """
    An experiment that trains and evaluates a machine learning model using the scikit-learn framework.
    """

    PredictorFactory = Callable[[], Predictor]

    def __create_learner(self, parameters: ParameterDict) -> BaseEstimator:
        learner = clone(self.base_learner)

        if parameters:
            learner.set_params(**parameters)
            log.info('Successfully applied parameter setting: %s', parameters)

        return learner

    @staticmethod
    def __check_for_parameter_changes(expected_parameters: Dict[str, Any], actual_parameters: Dict[str, Any]):
        changes = []

        for key, expected_value in expected_parameters.items():
            expected_value = str(expected_value)
            actual_value = str(actual_parameters[key])

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

    def __train(self, learner: BaseEstimator, dataset: Dataset) -> Timer.Duration:
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
                return self.__train(learner, dataset.enforce_dense_features())
            if dataset.has_sparse_outputs:
                return self.__train(learner, dataset.enforce_dense_outputs())
            raise error

    def __init__(self,
                 problem_type: ProblemType,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 dataset_splitter: DatasetSplitter,
                 prediction_output_writers: List[OutputWriter],
                 predictor_factory: PredictorFactory,
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
        :param predictor_factory:               A `PredictorFactory`
        :param fit_kwargs:                      Optional keyword arguments to be passed to the learner when fitting a
                                                model
        :param predict_kwargs:                  Optional keyword arguments to be passed to the learner when obtaining
                                                predictions from a model
        """
        super().__init__(problem_type=problem_type,
                         learner_name=learner_name,
                         dataset_splitter=dataset_splitter,
                         prediction_output_writers=prediction_output_writers)
        self.base_learner = base_learner
        self.predictor_factory = predictor_factory
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

    def _train(self, learner: Optional[Any], parameters: ParameterDict, dataset: Dataset) -> TrainingState:
        new_learner = self.__create_learner(parameters=parameters)

        # Use existing model, if possible, otherwise train a new model...
        if isinstance(learner, type(new_learner)):
            self.__check_for_parameter_changes(expected_parameters=parameters, actual_parameters=learner.get_params())
            return TrainingState(learner=learner)

        log.info('Fitting model to %s training examples...', dataset.num_examples)
        training_duration = self.__train(new_learner, dataset)
        log.info('Successfully fit model in %s', training_duration)
        return TrainingState(learner=new_learner, training_duration=training_duration)

    def _predict(self, learner: Any, dataset: Dataset) -> Generator[PredictionState]:
        predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
        predictor = self.predictor_factory()

        try:
            return predictor.obtain_predictions(learner, dataset, **predict_kwargs)
        except ValueError as error:
            if dataset.has_sparse_features:
                return self._predict(learner, dataset.enforce_dense_features())

            raise error
