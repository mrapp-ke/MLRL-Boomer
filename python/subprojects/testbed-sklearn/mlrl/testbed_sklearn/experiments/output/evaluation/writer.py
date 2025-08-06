"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results to one or several sinks.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from typing import Any, List, Optional, override

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class EvaluationDataExtractor(DataExtractor, ABC):
    """
    An abstract base class for all classes that allow obtaining evaluation results according to different evaluation
    measures.
    """

    def __init__(self):
        self.measurements = {}

    @override
    def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> Optional[OutputData]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        training_result = state.training_result
        prediction_result = state.prediction_result
        dataset_type = state.dataset_type
        dataset = state.dataset
        folding_strategy = state.folding_strategy

        if training_result and prediction_result and dataset_type and dataset and folding_strategy:
            measurements = self.measurements.setdefault(dataset_type, Measurements(folding_strategy.num_folds))
            fold_index = state.fold.index
            options = Options(reduce(lambda aggr, sink: aggr | sink.options.dictionary, sinks, {}))
            training_duration = training_result.training_duration.value
            prediction_duration = prediction_result.prediction_duration.value
            measurements.values_by_measure(EVALUATION_MEASURE_TRAINING_TIME)[fold_index] = training_duration
            measurements.values_by_measure(EVALUATION_MEASURE_PREDICTION_TIME)[fold_index] = prediction_duration
            self._update_measurements(measurements,
                                      fold_index,
                                      ground_truth=dataset.y,
                                      predictions=prediction_result.predictions,
                                      options=options)
            return EvaluationResult(measurements)

        return None

    @abstractmethod
    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any,
                             options: Options):
        """
        Must be implemented by subclasses in order to evaluate predictions and update a specific data point of given
        `Measurements` accordingly.

        :param measurements:    The `Measurements` to be updated
        :param index:           The index of the data point to be updated
        :param ground_truth:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth
        :param predictions:     A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the predictions to be evaluated
        :param options:         Options to be taken into account
        """


class EvaluationWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow writing evaluation results to one or several sinks.
    """

    @override
    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        fold = state.fold
        sink.write_to_sink(state, output_data, **{EvaluationResult.KWARG_FOLD: fold.index})
        folding_strategy = state.folding_strategy

        if folding_strategy and \
                folding_strategy.is_cross_validation_used and \
                not folding_strategy.is_subset and \
                folding_strategy.is_last_fold(fold):
            new_state = replace(state, fold=None)
            sink.write_to_sink(new_state, output_data)
