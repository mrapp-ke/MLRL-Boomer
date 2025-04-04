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


class EvaluationWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow writing evaluation results to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        self.measurements = {}

    @abstractmethod
    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any):
        """
        Must be implemented by subclasses in order to evaluate predictions and update a specific data point of given
        `Measurements` accordingly.

        :param measurements:    The `Measurements` to be updated
        :param index:           The index of the data point to be updated
        :param ground_truth:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth
        :param predictions:     A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the predictions to be evaluated
        """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        training_result = state.training_result
        prediction_result = state.prediction_result

        if training_result and prediction_result:
            fold = state.fold
            dataset = state.dataset
            data_type = dataset.type
            measurements = self.measurements.setdefault(data_type, Measurements(fold.num_folds))
            index = 0 if fold.index is None else fold.index
            training_duration = training_result.training_duration.value
            prediction_duration = prediction_result.prediction_duration.value
            measurements.values_by_measure(EVALUATION_MEASURE_TRAINING_TIME)[index] = training_duration
            measurements.values_by_measure(EVALUATION_MEASURE_PREDICTION_TIME)[index] = prediction_duration
            self._update_measurements(measurements,
                                      index,
                                      ground_truth=dataset.y,
                                      predictions=prediction_result.predictions)
            return EvaluationResult(measurements)

        return None

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        fold = state.fold
        fold_index = fold.index if fold.is_cross_validation_used else 0
        sink.write_to_sink(state, output_data, **{EvaluationResult.KWARG_FOLD: fold_index})

        if fold.is_cross_validation_used and fold.is_last_fold:
            sink.write_to_sink(replace(state, fold=replace(fold, index=None)), output_data)
