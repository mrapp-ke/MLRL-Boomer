"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for repeatedly obtaining predictions from an ensemble model, using only a subset of the ensemble
members.
"""
import logging as log

from typing import List

from mlrl.common.mixins import IncrementalClassifierMixin, IncrementalRegressorMixin

from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.prediction.predictor import Predictor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import ExperimentState, PredictionState
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.prediction_scope import IncrementalPrediction


class IncrementalPredictor(Predictor):
    """
    Repeatedly obtains predictions from a previously trained ensemble model, e.g., a model consisting of several rules,
    using only a subset of the ensemble members with increasing size.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter], min_size: int,
                 max_size: int, step_size: int):
        """
        :param min_size:    The minimum number of ensemble members to be evaluated. Must be at least 0
        :param max_size:    The maximum number of ensemble members to be evaluated. Must be greater than `min_size` or
                            0, if all ensemble members should be evaluated
        :param step_size:   The number of additional ensemble members to be considered at each repetition. Must be at
                            least 1
        """
        super().__init__(prediction_type, output_writers)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size

    def predict_and_evaluate(self, state: ExperimentState, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.prediction.predictor.Predictor.predict_and_evaluate`
        """
        learner = state.training_result.learner

        if not isinstance(learner, IncrementalClassifierMixin) and not isinstance(learner, IncrementalRegressorMixin):
            raise ValueError('Cannot obtain incremental predictions from a model of type ' + type(learner.__name__))

        predict_proba_function = learner.predict_proba_incrementally if callable(
            getattr(learner, 'predict_proba_incrementally', None)) else None
        dataset = state.dataset
        incremental_predictor = self._invoke_prediction_function(learner, learner.predict_incrementally,
                                                                 predict_proba_function, dataset, **kwargs)

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
                         dataset.type.value, current_size)
                start_time = Timer.start()
                predictions = incremental_predictor.apply_next(next_step_size)
                prediction_duration = Timer.stop(start_time)

                if predictions is not None:
                    log.info('Successfully predicted in %s', prediction_duration)
                    state.prediction_result = PredictionState(predictions=predictions,
                                                              prediction_type=self.prediction_type,
                                                              prediction_scope=IncrementalPrediction(current_size),
                                                              prediction_duration=prediction_duration)
                    self._evaluate_predictions(state)

                next_step_size = step_size
                current_size = min(current_size + next_step_size, total_size)
