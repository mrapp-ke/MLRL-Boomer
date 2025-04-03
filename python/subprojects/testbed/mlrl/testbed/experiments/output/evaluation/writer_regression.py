"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to regression evaluation measures to one or several
sinks.
"""
from typing import Any

from mlrl.common.data.arrays import enforce_dense
from mlrl.common.data.types import Float32

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.evaluation.measures_regression import REGRESSION_EVALUATION_MEASURES
from mlrl.testbed.experiments.output.evaluation.writer import EvaluationWriter
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.fold import Fold


class RegressionEvaluationWriter(EvaluationWriter):
    """
    Allows writing evaluation results according to regression evaluation measures to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.regression_evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, *options)

    def _update_measurements(self, measurements: Measurements, ground_truth: Any, predictions: Any, fold: Fold):
        ground_truth = enforce_dense(ground_truth, order='C', dtype=Float32)
        evaluation_measures = self.regression_evaluation_measures

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.put(evaluation_measure, value, num_folds=fold.num_folds, fold=fold.index)
