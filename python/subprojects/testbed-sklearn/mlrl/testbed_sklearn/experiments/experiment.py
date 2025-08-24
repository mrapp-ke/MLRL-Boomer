"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments using the scikit-learn framework.
"""
import logging as log

from dataclasses import replace
from functools import reduce
from typing import Any, Dict, Generator, Optional, override

from sklearn.base import BaseEstimator, clone

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.characteristics.data.writer_data import DataCharacteristicsWriter
from mlrl.testbed_sklearn.experiments.output.characteristics.data.writer_prediction import \
    PredictionCharacteristicsWriter
from mlrl.testbed_sklearn.experiments.output.dataset.writer_ground_truth import GroundTruthWriter
from mlrl.testbed_sklearn.experiments.output.dataset.writer_prediction import PredictionWriter
from mlrl.testbed_sklearn.experiments.output.evaluation.writer import EvaluationWriter
from mlrl.testbed_sklearn.experiments.output.label_vectors import LabelVectorWriter

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.state import ExperimentState, ParameterDict, PredictionState, TrainingState
from mlrl.testbed.experiments.timer import Timer


class SkLearnExperiment(Experiment):
    """
    An experiment that trains and evaluates a machine learning model using the scikit-learn framework.
    """

    class Builder(Experiment.Builder):
        """
        Allows to configure and create instances of the class `SkLearnExperiment`.
        """

        def __init__(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter):
            """
            :param initial_state:       The initial state of the experiment
            :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
            """
            super().__init__(initial_state=initial_state, dataset_splitter=dataset_splitter)
            self.data_characteristics_writer = DataCharacteristicsWriter()
            self.prediction_characteristics_writer = PredictionCharacteristicsWriter()
            self.ground_truth_writer = GroundTruthWriter()
            self.prediction_writer = PredictionWriter()
            self.label_vector_writer = LabelVectorWriter()
            self.evaluation_writer = EvaluationWriter()
            self.add_pre_training_output_writers(self.data_characteristics_writer)
            self.add_post_training_output_writers(self.label_vector_writer)
            self.add_prediction_output_writers(
                self.prediction_characteristics_writer,
                self.ground_truth_writer,
                self.prediction_writer,
                self.evaluation_writer,
            )

        @override
        def _create_experiment(self, initial_state: ExperimentState, dataset_splitter: DatasetSplitter) -> Experiment:
            return SkLearnExperiment(initial_state=initial_state, dataset_splitter=dataset_splitter)

    def __create_learner(self, parameters: ParameterDict) -> BaseEstimator:
        learner = clone(self.initial_state.problem_domain.base_learner)

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

    def _fit(self, estimator: BaseEstimator, dataset: TabularDataset,
             fit_kwargs: Optional[Dict[str, Any]]) -> Timer.Duration:
        """
        May be overridden by subclasses in order to fit a scikit-learn estimator to a dataset.

        :param estimator:   A scikit-learn estimator
        :param fit_kwargs:  Optional keyword arguments to be passed to the estimator

        """
        fit_kwargs = fit_kwargs if fit_kwargs else {}

        try:
            start_time = Timer.start()
            estimator.fit(dataset.x, dataset.y, **fit_kwargs)
            return Timer.stop(start_time)
        except ValueError as error:
            if dataset.has_sparse_features:
                return self._fit(estimator, dataset.enforce_dense_features(), fit_kwargs)
            if dataset.has_sparse_outputs:
                return self._fit(estimator, dataset.enforce_dense_outputs(), fit_kwargs)
            raise error

    @override
    def _train(self, learner: Optional[Any], parameters: ParameterDict, dataset: Dataset) -> TrainingState:
        new_learner = self.__create_learner(parameters=parameters)

        # Use existing model, if possible, otherwise train a new model...
        if isinstance(learner, type(new_learner)):
            self.__check_for_parameter_changes(expected_parameters=parameters, actual_parameters=learner.get_params())
            return TrainingState(learner=learner)

        log.info('Fitting model to %s training examples...', dataset.num_examples)
        training_duration = self._fit(new_learner, dataset, fit_kwargs=self.initial_state.problem_domain.fit_kwargs)
        log.info('Successfully fit model in %s', training_duration)
        return TrainingState(learner=new_learner, training_duration=training_duration)

    @override
    def _predict(self, state: ExperimentState) -> Generator[PredictionState, None, None]:
        dataset = state.dataset_as(self, TabularDataset)
        learner = state.learner_as(self, BaseEstimator)

        if dataset and learner:
            try:
                problem_domain = self.initial_state.problem_domain
                predict_kwargs = problem_domain.predict_kwargs
                predict_kwargs = predict_kwargs if predict_kwargs else {}
                predictor = problem_domain.predictor_factory.create()
                dataset_type = state.dataset_type
                yield from predictor.obtain_predictions(learner, dataset, dataset_type, **predict_kwargs)
            except ValueError as error:
                if dataset.has_sparse_features:
                    yield self._predict(replace(state, dataset=dataset.enforce_dense_features()))

                raise error
