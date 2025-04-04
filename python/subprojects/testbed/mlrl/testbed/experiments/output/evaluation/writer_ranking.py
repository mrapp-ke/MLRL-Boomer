"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to ranking evaluation measures to one or several sinks.
"""
from typing import Any

from sklearn.utils.multiclass import is_multilabel

from mlrl.common.data.arrays import enforce_dense
from mlrl.common.data.types import Uint8

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.evaluation.measures_ranking import RANKING_EVALUATION_MEASURES
from mlrl.testbed.experiments.output.evaluation.measures_regression import REGRESSION_EVALUATION_MEASURES
from mlrl.testbed.experiments.output.evaluation.writer import EvaluationWriter
from mlrl.testbed.experiments.output.sinks import Sink


class RankingEvaluationWriter(EvaluationWriter):
    """
    Allows writing evaluation results according to ranking evaluation measures to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.regression_evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, *options)
        self.ranking_evaluation_measures = OutputValue.filter_values(RANKING_EVALUATION_MEASURES, *options)

    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any):
        ground_truth = enforce_dense(ground_truth, order='C', dtype=Uint8)

        if is_multilabel(ground_truth):
            evaluation_measures = self.ranking_evaluation_measures + self.regression_evaluation_measures
        else:
            evaluation_measures = self.regression_evaluation_measures

            if predictions.shape[1] > 1:
                predictions = predictions[:, -1]

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.values_by_measure(evaluation_measure)[index] = value
