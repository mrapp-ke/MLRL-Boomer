"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for repeatedly obtaining predictions from an ensemble model, using only a subset of the ensemble
members.
"""
import logging as log

from typing import Any, Generator, override

from sklearn.base import BaseEstimator

from mlrl.common.mixins import IncrementalClassifierMixin, IncrementalRegressorMixin

from mlrl.testbed_sklearn.experiments.prediction.predictor import PredictionFunction, Predictor

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import PredictionState
from mlrl.testbed.experiments.timer import Timer


class IncrementalPredictionFunction(PredictionFunction):
    """
    A function that obtains and returns incremental predictions from a learner.
    """

    def __init__(self, learner: BaseEstimator):
        super().__init__(learner=learner,
                         predict_function=learner.predict_incrementally,
                         predict_proba_function=learner.predict_proba_incrementally if callable(
                             getattr(learner, 'predict_proba_incrementally', None)) else None)


class IncrementalPredictor(Predictor):
    """
    Repeatedly obtains predictions from a previously trained ensemble model, e.g., a model consisting of several rules,
    using only a subset of the ensemble members with increasing size.
    """

    class Scope(PredictionScope):
        """
        Provides information about predictions that have been obtained incrementally.
        """

        def __init__(self, model_size: int):
            """
            :param model_size: The size of the model, the predictions have been obtained from
            """
            self._model_size = model_size

        @override
        @property
        def model_size(self) -> int:
            """
            See :func:`mlrl.testbed.prediction_scope.PredictionScope.model_size`
            """
            return self._model_size

    def __init__(self, prediction_type: PredictionType, min_size: int, max_size: int, step_size: int):
        """
        :param min_size:    The minimum number of ensemble members to be evaluated. Must be at least 0
        :param max_size:    The maximum number of ensemble members to be evaluated. Must be greater than `min_size` or
                            0, if all ensemble members should be evaluated
        :param step_size:   The number of additional ensemble members to be considered at each repetition. Must be at
                            least 1
        """
        super().__init__(prediction_type)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size

    @override
    def obtain_predictions(self, learner: Any, dataset: Dataset, dataset_type: DatasetType,
                           **kwargs) -> Generator[PredictionState, None, None]:
        """
        See :func:`mlrl.testbed_sklearn.experiments.prediction.predictor.Predictor.obtain_predictions`
        """
        if not isinstance(learner, IncrementalClassifierMixin) and not isinstance(learner, IncrementalRegressorMixin):
            raise ValueError('Cannot obtain incremental predictions from a model of type ' + type(learner.__name__))

        prediction_function = IncrementalPredictionFunction(learner)
        incremental_predictor = prediction_function.invoke(dataset, self.prediction_type, **kwargs)

        if incremental_predictor:
            step_size = self.step_size
            total_size = incremental_predictor.get_num_next()
            max_size = self.max_size

            if max_size > 0:
                total_size = min(max_size, total_size)

            min_size = self.min_size
            next_step_size = min_size if min_size > 0 else step_size
            current_size = min(next_step_size, total_size)

            while incremental_predictor.has_next():
                log.info('Predicting for %s %s examples using a model of size %s...', dataset.num_examples,
                         dataset_type, current_size)
                start_time = Timer.start()
                predictions = incremental_predictor.apply_next(next_step_size)
                prediction_duration = Timer.stop(start_time)

                if predictions is not None:
                    log.info('Successfully predicted in %s', prediction_duration)
                    yield PredictionState(predictions=predictions,
                                          prediction_type=self.prediction_type,
                                          prediction_scope=IncrementalPredictor.Scope(current_size),
                                          prediction_duration=prediction_duration)

                next_step_size = step_size
                current_size = min(current_size + next_step_size, total_size)
