"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results to one or several sinks.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import reduce
from itertools import chain
from typing import Any, Dict, List, Optional, override

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult
from mlrl.testbed_sklearn.experiments.output.evaluation.measures_classification import \
    MULTI_LABEL_EVALUATION_MEASURES, SINGLE_LABEL_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.measures_ranking import RANKING_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.measures_regression import REGRESSION_EVALUATION_MEASURES

from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import parse_number

from mlrl.util.options import Options


class EvaluationDataExtractor(DataExtractor, ABC):
    """
    An abstract base class for all classes that allow obtaining evaluation results according to different evaluation
    measures.
    """

    measurements: Dict[DatasetType, Measurements] = {}

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
        fold = state.fold

        if training_result and prediction_result and dataset_type and dataset and folding_strategy:
            measurements = self.measurements.setdefault(dataset_type, Measurements(folding_strategy.num_folds))

            if fold:
                fold_index = fold.index
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


class EvaluationWriter(ResultWriter):
    """
    Allows writing evaluation results to one or several sinks.
    """

    class InputExtractor(TabularDataExtractor):
        """
        Uses `TabularInputData` that has previously been loaded via an input reader.
        """

        ALL_MEASURES = set(
            chain(MULTI_LABEL_EVALUATION_MEASURES, SINGLE_LABEL_EVALUATION_MEASURES, RANKING_EVALUATION_MEASURES,
                  REGRESSION_EVALUATION_MEASURES))

        measurements: Dict[DatasetType, Measurements] = {}

        @override
        def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset_type = state.dataset_type
            folding_strategy = state.folding_strategy

            if dataset_type and folding_strategy:
                measurements = self.measurements.setdefault(dataset_type, Measurements(folding_strategy.num_folds))
                tabular_output_data = super().extract_data(state, sinks)
                fold = state.fold

                if tabular_output_data and fold:
                    table = tabular_output_data.to_table(Options()).to_column_wise_table()
                    columns_by_name = {column.header: column for column in table.columns}

                    for measure in chain(self.ALL_MEASURES, map(Measure.std_dev, self.ALL_MEASURES)):
                        column = columns_by_name.get(measure.name)

                        if column:
                            values = measurements.values_by_measure(measure)
                            values[fold.index] = parse_number(column[0], percentage=measure.percentage)

                return EvaluationResult(measurements) if measurements else None

            return None

    def __init__(self, *extractors: EvaluationDataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors,
                         input_data=TabularInputData(properties=EvaluationResult.PROPERTIES,
                                                     context=EvaluationResult.CONTEXT))

    @override
    def _create_states(self, state: ExperimentState) -> List[ExperimentState]:
        states = super()._create_states(state)
        folding_strategy = state.folding_strategy

        if folding_strategy and \
                folding_strategy.is_cross_validation_used and \
                not folding_strategy.is_subset and \
                folding_strategy.is_last_fold(state.fold):
            states.append(replace(state, fold=None))

        return states

    @override
    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        fold = state.fold
        kwargs = {EvaluationResult.KWARG_FOLD: fold.index} if fold else {}
        sink.write_to_sink(state, output_data, **kwargs)
