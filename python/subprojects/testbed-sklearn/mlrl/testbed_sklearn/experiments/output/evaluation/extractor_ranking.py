"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to ranking evaluation measures to one or several sinks.
"""
from itertools import chain
from typing import Any, override

from sklearn.utils.multiclass import is_multilabel

from mlrl.testbed_sklearn.experiments.output.evaluation.measures_ranking import RANKING_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.measures_regression import REGRESSION_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.writer import EvaluationDataExtractor

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure

from mlrl.util.arrays import enforce_dense
from mlrl.util.options import Options


class RankingEvaluationDataExtractor(EvaluationDataExtractor):
    """
    Obtains evaluation results according to ranking evaluation measures.
    """

    @override
    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any,
                             options: Options):
        ground_truth = enforce_dense(ground_truth, order='C')
        regression_evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, options)

        if is_multilabel(ground_truth):
            ranking_evaluation_measures = OutputValue.filter_values(RANKING_EVALUATION_MEASURES, options)
            evaluation_measures = chain(ranking_evaluation_measures, regression_evaluation_measures)
        else:
            evaluation_measures = regression_evaluation_measures

            if predictions.shape[1] > 1:
                predictions = predictions[:, -1]

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.values_by_measure(evaluation_measure)[index] = value
