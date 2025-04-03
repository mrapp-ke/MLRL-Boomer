"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing evaluation results according to different measures that are part of output data.
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import OutputValue, TabularOutputData
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

    def __init__(self):
        super().__init__('Evaluation result', 'evaluation')
        self.get_formatter_options(CsvFileSink).include_prediction_scope = False
        self.measures: Set[OutputValue] = set()
        self.results: Optional[List[Dict[OutputValue, float]]] = None

    def put(self, measure: OutputValue, score: float, num_folds: int, fold: Optional[int]):
        """
        Adds a new score according to a specific measure to the evaluation result.

        :param measure:     The measure
        :param score:       The score according to the measure
        :param num_folds:   The total number of cross validation folds
        :param fold:        The fold, the score corresponds to, or None, if no cross validation is used
        """
        results = self.results

        if not results:
            results = [{} for _ in range(num_folds)]
            self.results = results

        if len(results) != num_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(measure)
        values = results[0 if fold is None else fold]
        values[measure] = score

    def get(self, measure: OutputValue, fold: Optional[int], **kwargs) -> str:
        """
        Returns the score according to a specific measure.

        :param measure: The measure
        :param fold:    The fold, the score corresponds to, or None, if no cross validation is used
        :return:        A textual representation of the score
        """
        results = self.results

        if not results:
            raise AssertionError('No evaluation results available')

        score = results[0 if fold is None else fold][measure]
        return measure.format(score, **kwargs)

    def dict(self, fold: Optional[int], **kwargs) -> Dict[OutputValue, str]:
        """
        Returns a dictionary that stores the scores for a specific fold according to each measure.

        :param fold:    The fold, the scores correspond to, or None, if no cross validation is used
        :return:        A dictionary that stores textual representations of the scores for the given fold according
                        to each measure
        """
        results = self.results

        if not results:
            raise AssertionError('No evaluation results available')

        result_dict = {}

        for measure, score in results[0 if fold is None else fold].items():
            result_dict[measure] = measure.format(score, **kwargs)

        return result_dict

    def avg(self, measure: OutputValue, **kwargs) -> Tuple[str, str]:
        """
        Returns the score and standard deviation according to a specific measure averaged over all available folds.

        :param measure: The measure
        :return:        A tuple consisting of textual representations of the averaged score and standard deviation
        """
        values = [results[measure] for results in self.results if results]
        values = np.array(values)
        return measure.format(np.average(values), **kwargs), measure.format(np.std(values), **kwargs)

    def avg_dict(self, **kwargs) -> Dict[OutputValue, str]:
        """
        Returns a dictionary that stores the scores, averaged across all folds, as well as the standard deviation,
        according to each measure.

        :return: A dictionary that stores textual representations of the scores and standard deviation according to
                 each measure
        """
        result: Dict[OutputValue, str] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure, **kwargs)
            result[measure] = score
            result[OutputValue(measure.option_key, 'Std.-dev. ' + measure.name, measure.percentage)] = std_dev

        return result

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        fold = kwargs.get(self.KWARG_FOLD)
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        enable_all = options.get_bool(self.OPTION_ENABLE_ALL, True)
        rows = []

        for measure in sorted(self.measures):
            if options.get_bool(measure.option_key, enable_all) and measure.option_key != self.OPTION_TRAINING_TIME \
                    and measure.option_key != self.OPTION_PREDICTION_TIME:
                if fold is None:
                    score, std_dev = self.avg(measure, percentage=percentage, decimals=decimals)
                    rows.append([str(measure), score, '±' + std_dev])
                else:
                    score = self.get(measure, fold, percentage=percentage, decimals=decimals)
                    rows.append([str(measure), score])

        return format_table(rows)

    def to_table(self, options: Options, **kwargs) -> Optional[TabularOutputData.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
        """
        fold = kwargs.get(self.KWARG_FOLD)
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 0)
        enable_all = options.get_bool(self.OPTION_ENABLE_ALL, True)

        if fold is None:
            columns = self.avg_dict(percentage=percentage, decimals=decimals)
        else:
            columns = self.dict(fold, percentage=percentage, decimals=decimals)

        filtered_columns = {}

        for measure, value in columns.items():
            if options.get_bool(measure.option_key, enable_all):
                filtered_columns[measure.name] = value

        return [filtered_columns]


EVALUATION_MEASURE_TRAINING_TIME = OutputValue(EvaluationResult.OPTION_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = OutputValue(EvaluationResult.OPTION_PREDICTION_TIME, 'Prediction Time')
