"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing evaluation results that are part of output data.
"""
from itertools import tee
from typing import Any, Dict, List, Optional, Tuple, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import TabularProperties
from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.output.evaluation.evaluation_result import AggregatedEvaluationResult
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.sinks import CsvFileSink
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.options import Options


class EvaluationResult(TabularOutputData):
    """
    Stores the evaluation results according to different measures.
    """

    PROPERTIES = TabularProperties(name='Evaluation result', file_name='evaluation')

    CONTEXT = Context()

    OPTION_HAMMING_LOSS = 'hamming_loss'

    OPTION_HAMMING_ACCURACY = 'hamming_accuracy'

    OPTION_SUBSET_ZERO_ONE_LOSS = 'subset_zero_one_loss'

    OPTION_SUBSET_ACCURACY = 'subset_accuracy'

    OPTION_MICRO_PRECISION = 'micro_precision'

    OPTION_MICRO_RECALL = 'micro_recall'

    OPTION_MICRO_F1 = 'micro_f1'

    OPTION_MICRO_JACCARD = 'micro_jaccard'

    OPTION_MACRO_PRECISION = 'macro_precision'

    OPTION_MACRO_RECALL = 'macro_recall'

    OPTION_MACRO_F1 = 'macro_f1'

    OPTION_MACRO_JACCARD = 'macro_jaccard'

    OPTION_EXAMPLE_WISE_PRECISION = 'example_wise_precision'

    OPTION_EXAMPLE_WISE_RECALL = 'example_wise_recall'

    OPTION_EXAMPLE_WISE_F1 = 'example_wise_f1'

    OPTION_EXAMPLE_WISE_JACCARD = 'example_wise_jaccard'

    OPTION_ACCURACY = 'accuracy'

    OPTION_ZERO_ONE_LOSS = 'zero_one_loss'

    OPTION_PRECISION = 'precision'

    OPTION_RECALL = 'recall'

    OPTION_F1 = 'f1'

    OPTION_JACCARD = 'jaccard'

    OPTION_MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'

    OPTION_MEAN_SQUARED_ERROR = 'mean_squared_error'

    OPTION_MEDIAN_ABSOLUTE_ERROR = 'mean_absolute_error'

    OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'

    OPTION_RANK_LOSS = 'rank_loss'

    OPTION_COVERAGE_ERROR = 'coverage_error'

    OPTION_LABEL_RANKING_AVERAGE_PRECISION = 'lrap'

    OPTION_DISCOUNTED_CUMULATIVE_GAIN = 'dcg'

    OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 'ndcg'

    OPTION_TRAINING_TIME = 'training_time'

    OPTION_PREDICTION_TIME = 'prediction_time'

    KWARG_FOLD = 'fold_index'

    def __init__(self, measurements: Measurements):
        """
        :param measurements: The measurements according to different evaluation measures
        """
        super().__init__(properties=self.PROPERTIES, context=self.CONTEXT)
        self.get_context(CsvFileSink).include_prediction_scope = False
        self.measurements = measurements

    def __get_unformatted_header(self, header: Any) -> str:
        return str(header).rstrip('(↑)').rstrip('(↓)').rstrip()

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        table = self.to_table(options, **kwargs)

        if table:
            header_row = table.header_row
            fold = kwargs.get(self.KWARG_FOLD)
            variants_by_measure: Dict[str, List[Tuple[str, int]]] = {}

            for column_index in range(0, table.num_columns, 2 if fold is None else 1):
                header = header_row[column_index] if header_row else None

                if not header or header.option_key not in {self.OPTION_TRAINING_TIME, self.OPTION_PREDICTION_TIME}:
                    measure_name = self.__get_unformatted_header(header)
                    at_index = measure_name.find('@')

                    if at_index >= 0:
                        measure_name = measure_name[:at_index]

                    variants_by_measure.setdefault(measure_name, []).append((measure_name, column_index))

            first_row = next(table.rows)
            rotated_table = RowWiseTable()

            for measure_name, variants in variants_by_measure.items():
                new_row: List[Any] = []

                for _, column_index in sorted(variants, key=lambda x: x[0]):
                    header = header_row[column_index] if header_row else None
                    new_row.append(self.__get_unformatted_header(header) + ':' if header and new_row else header)
                    new_row.append(first_row[column_index])

                    if fold is None:
                        new_row.append('±' + first_row[column_index + 1])

                rotated_table.add_row(*new_row)

            return rotated_table.sort_by_columns(0).format()

        return None

    @override
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        percentage = options.get_bool(OPTION_PERCENTAGE, kwargs.get(OPTION_PERCENTAGE, True))
        decimals = options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0))
        enable_all = options.get_bool(AggregatedEvaluationResult.OPTION_ENABLE_ALL,
                                      kwargs.get(AggregatedEvaluationResult.OPTION_ENABLE_ALL, True))
        fold = kwargs.get(self.KWARG_FOLD)
        dictionary = self.measurements.averages_as_dict() if fold is None else self.measurements.values_as_dict(fold)
        headers, measures = tee(
            filter(lambda measure: options.get_bool(measure.option_key, enable_all), dictionary.keys()))
        values = map(lambda measure: measure.format(dictionary[measure], percentage=percentage, decimals=decimals),
                     measures)
        return RowWiseTable(*headers).add_row(*values)


EVALUATION_MEASURE_TRAINING_TIME = OutputValue(
    option_key=EvaluationResult.OPTION_TRAINING_TIME,
    name='Training Time (' + Measure.UNIT_SECONDS + ')',
)

EVALUATION_MEASURE_PREDICTION_TIME = OutputValue(
    option_key=EvaluationResult.OPTION_PREDICTION_TIME,
    name='Prediction Time (' + Measure.UNIT_SECONDS + ')',
)
