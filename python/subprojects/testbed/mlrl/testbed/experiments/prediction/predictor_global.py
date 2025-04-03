"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from global machine learning models.
"""
import logging as log

from mlrl.testbed.experiments.prediction.predictor import Predictor
from mlrl.testbed.experiments.state import ExperimentState, PredictionState
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.prediction_scope import GlobalPrediction


class GlobalPredictor(Predictor):
    """
    Obtains predictions from a previously trained global model.
    """

    def predict_and_evaluate(self, state: ExperimentState, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.prediction.predictor.Predictor.predict_and_evaluate`
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
            state.prediction_result = PredictionState(predictions=predictions,
                                                      prediction_type=self.prediction_type,
                                                      prediction_scope=GlobalPrediction(),
                                                      prediction_duration=prediction_duration)
            self._evaluate_predictions(state)
