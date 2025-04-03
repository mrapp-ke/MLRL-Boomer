"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results to one or several sinks.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Optional

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.fold import Fold


class EvaluationWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow writing evaluation results to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        self.measurements = {}

    @abstractmethod
    def _update_measurements(self, measurements: Measurements, ground_truth: Any, predictions: Any, fold: Fold):
        """
        Must be implemented by subclasses in order to evaluate predictions and update given `Measurements` accordingly.

        :param measurements:    The `Measurements` to be updated
        :param ground_truth:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth
        :param predictions:     A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the predictions to be evaluated
        :param fold:            The fold of the dataset, the given predictions and ground truth correspond to
        """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        training_result = state.training_result
        prediction_result = state.prediction_result

        if training_result and prediction_result:
            fold = state.fold
            dataset = state.dataset
            data_type = dataset.type
            measurements = self.measurements.setdefault(data_type, Measurements())
            measurements.put(EVALUATION_MEASURE_TRAINING_TIME,
                             training_result.training_duration.value,
                             num_folds=fold.num_folds,
                             fold=fold.index)
            measurements.put(EVALUATION_MEASURE_PREDICTION_TIME,
                             prediction_result.prediction_duration.value,
                             num_folds=fold.num_folds,
                             fold=fold.index)
            self._update_measurements(measurements,
                                      ground_truth=dataset.y,
                                      predictions=prediction_result.predictions,
                                      fold=fold)
            return EvaluationResult(measurements)

        return None

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        fold = state.fold
        fold_index = fold.index if fold.is_cross_validation_used else 0
        sink.write_to_sink(state, output_data, **{EvaluationResult.KWARG_FOLD: fold_index})

        if fold.is_cross_validation_used and fold.is_last_fold:
            sink.write_to_sink(replace(state, fold=replace(fold, index=None)), output_data)
