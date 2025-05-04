"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from functools import reduce
from typing import Any, Callable, Dict, Generator, Optional

from sklearn.base import BaseEstimator, clone

from mlrl.common.mixins import NominalFeatureSupportMixin, OrdinalFeatureSupportMixin

from mlrl.testbed.experiments.dataset import AttributeType, Dataset, DatasetType
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.prediction import Predictor
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ParameterDict, PredictionState, TrainingState
from mlrl.testbed.experiments.timer import Timer


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
                 predictor_factory: PredictorFactory,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param problem_type:        The type of the machine learning problem
        :param base_learner:        The machine learning algorithm to be used
        :param learner_name:        The name of the machine learning algorithm
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        :param predictor_factory:   A `PredictorFactory`
        :param fit_kwargs:          Optional keyword arguments to be passed to the learner when fitting a model
        :param predict_kwargs:      Optional keyword arguments to be passed to the learner when obtaining predictions
                                    from a model
        """
        super().__init__(problem_type=problem_type, learner_name=learner_name, dataset_splitter=dataset_splitter)
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

    def _predict(self, learner: Any, dataset: Dataset, dataset_type: DatasetType) -> Generator[PredictionState]:
        predict_kwargs = self.predict_kwargs if self.predict_kwargs else {}
        predictor = self.predictor_factory()

        try:
            return predictor.obtain_predictions(learner, dataset, dataset_type, **predict_kwargs)
        except ValueError as error:
            if dataset.has_sparse_features:
                return self._predict(learner, dataset.enforce_dense_features(), dataset_type)

            raise error
