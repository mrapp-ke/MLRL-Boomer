"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from global machine learning models.
"""
import logging as log

from typing import Generator

from sklearn.base import BaseEstimator

from mlrl.testbed.experiments.prediction.predictor import PredictionFunction, Predictor
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.state import ExperimentState, PredictionState
from mlrl.testbed.experiments.timer import Timer


class GlobalPredictionFunction(PredictionFunction):
    """
    A function that obtains and returns global predictions from a learner.
    """

    def __init__(self, learner: BaseEstimator):
        """
        :param learner: The learner, the predictions should be obtained from
        """
        super().__init__(
            learner=learner,
            predict_function=learner.predict,
            predict_proba_function=learner.predict_proba if callable(getattr(learner, 'predict_proba', None)) else None)


class GlobalPredictor(Predictor):
    """
    Obtains predictions from a previously trained global model.
    """

    class Scope(PredictionScope):
        """
        Provides information about predictions that have been obtained from a global model.
        """

        @property
        def model_size(self) -> int:
            """
            See :func:`mlrl.testbed.prediction_scope.PredictionScope.model_size`
            """
            return 0

    def obtain_predictions(self, state: ExperimentState, **kwargs) -> Generator[PredictionState]:
        """
        See :func:`mlrl.testbed.experiments.prediction.predictor.Predictor.obtain_predictions`
        """
        dataset = state.dataset
        log.info('Predicting for %s %s examples...', dataset.num_examples, state.dataset_type.value)
        learner = state.training_result.learner
        start_time = Timer.start()
        prediction_function = GlobalPredictionFunction(learner)
        predictions = prediction_function.invoke(dataset, self.prediction_type, **kwargs)
        prediction_duration = Timer.stop(start_time)

        if predictions is not None:
            log.info('Successfully predicted in %s', prediction_duration)
            yield PredictionState(predictions=predictions,
                                  prediction_type=self.prediction_type,
                                  prediction_scope=GlobalPredictor.Scope(),
                                  prediction_duration=prediction_duration)
