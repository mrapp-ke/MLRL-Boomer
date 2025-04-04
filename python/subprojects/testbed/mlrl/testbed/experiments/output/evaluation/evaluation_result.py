"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing evaluation results according to different measures that are part of output data.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.sinks import CsvFileSink
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_table


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
        super().__init__('Evaluation result', 'evaluation')
        self.get_formatter_options(CsvFileSink).include_prediction_scope = False
        self.measurements = measurements

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        measurements = self.measurements
        fold = kwargs.get(self.KWARG_FOLD)
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        enable_all = options.get_bool(self.OPTION_ENABLE_ALL, True)
        rows = []

        for measure in sorted(measurements.measures):
            if options.get_bool(measure.option_key, enable_all) and measure.option_key != self.OPTION_TRAINING_TIME \
                    and measure.option_key != self.OPTION_PREDICTION_TIME:
                if fold is None:
                    average, std_dev = measurements.average_by_measure(measure)
                    rows.append([
                        str(measure),
                        measure.format(average, percentage=percentage, decimals=decimals),
                        '±' + measure.format(std_dev, percentage=percentage, decimals=decimals),
                    ])
                else:
                    value = measurements.values_by_measure(measure)[fold]
                    rows.append([str(measure), measure.format(value, percentage=percentage, decimals=decimals)])

        return format_table(rows)

    def to_table(self, options: Options, **kwargs) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        measurements = self.measurements
        fold = kwargs.get(self.KWARG_FOLD)
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 0)
        enable_all = options.get_bool(self.OPTION_ENABLE_ALL, True)

        dictionary = measurements.averages_as_dict() if fold is None else measurements.values_as_dict(index=fold)
        columns = {
            measure: measure.format(value, percentage=percentage, decimals=decimals)
            for measure, value in dictionary.items()
        }

        filtered_columns = {}

        for measure, value in columns.items():
            if options.get_bool(measure.option_key, enable_all):
                filtered_columns[measure.name] = value

        return [filtered_columns]


EVALUATION_MEASURE_TRAINING_TIME = OutputValue(EvaluationResult.OPTION_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = OutputValue(EvaluationResult.OPTION_PREDICTION_TIME, 'Prediction Time')
