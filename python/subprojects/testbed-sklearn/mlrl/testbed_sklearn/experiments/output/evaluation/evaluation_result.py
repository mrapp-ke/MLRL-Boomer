"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing evaluation results that are part of output data.
"""
from itertools import tee
from typing import Optional, override

from mlrl.testbed.experiments.output.data import OutputData, OutputValue, TabularOutputData
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.sinks import CsvFileSink
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.options import Options


class EvaluationResult(TabularOutputData):
    """
    Stores the evaluation results according to different measures.
    """

    OPTION_ENABLE_ALL = 'enable_all'

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
        super().__init__(OutputData.Properties(name='Evaluation result', file_name='evaluation'))
        self.get_context(CsvFileSink).include_prediction_scope = False
        self.measurements = measurements

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        kwargs = dict(kwargs) | {OPTION_DECIMALS: 2}
        table = self.to_table(options, **kwargs)

        if table:
            header_row = table.header_row
            first_row = next(table.rows)
            fold = kwargs.get(self.KWARG_FOLD)
            rotated_table = RowWiseTable()

            for column_index in range(0, table.num_columns, 2 if fold is None else 1):
                header = header_row[column_index] if header_row else None

                if not header or header.option_key not in {self.OPTION_TRAINING_TIME, self.OPTION_PREDICTION_TIME}:
                    value = first_row[column_index]
                    new_row = [header, value] if header else [value]

                    if fold is None:
                        std_dev = '±' + first_row[column_index + 1]
                        new_row.append(std_dev)

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
        enable_all = options.get_bool(self.OPTION_ENABLE_ALL, kwargs.get(self.OPTION_ENABLE_ALL, True))
        fold = kwargs.get(self.KWARG_FOLD)
        dictionary = self.measurements.averages_as_dict() if fold is None else self.measurements.values_as_dict(fold)
        headers, measures = tee(
            filter(lambda measure: options.get_bool(measure.option_key, enable_all), dictionary.keys()))
        values = map(lambda measure: measure.format(dictionary[measure], percentage=percentage, decimals=decimals),
                     measures)
        return RowWiseTable(*headers).add_row(*values)


EVALUATION_MEASURE_TRAINING_TIME = OutputValue(EvaluationResult.OPTION_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = OutputValue(EvaluationResult.OPTION_PREDICTION_TIME, 'Prediction Time')
