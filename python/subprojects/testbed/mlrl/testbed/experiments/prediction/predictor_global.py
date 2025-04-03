"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from global machine learning models.
"""
import logging as log

from typing import Generator

from mlrl.testbed.experiments.prediction.predictor import Predictor
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.state import ExperimentState, PredictionState
from mlrl.testbed.experiments.timer import Timer


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
        log.info('Predicting for %s %s examples...', dataset.num_examples, dataset.type.value)
        learner = state.training_result.learner
        start_time = Timer.start()
        predict_proba_function = learner.predict_proba if callable(getattr(learner, 'predict_proba', None)) else None
        predictions = self._invoke_prediction_function(learner, learner.predict, predict_proba_function, dataset,
                                                       **kwargs)
        prediction_duration = Timer.stop(start_time)

        if predictions is not None:
            log.info('Successfully predicted in %s', prediction_duration)
            yield PredictionState(predictions=predictions,
                                  prediction_type=self.prediction_type,
                                  prediction_scope=GlobalPredictor.Scope(),
                                  prediction_duration=prediction_duration)
