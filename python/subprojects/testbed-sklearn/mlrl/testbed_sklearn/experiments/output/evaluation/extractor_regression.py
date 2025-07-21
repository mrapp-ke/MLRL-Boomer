"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to regression evaluation measures to one or several
sinks.
"""
from typing import Any, override

from mlrl.testbed_sklearn.experiments.output.evaluation.measures_regression import REGRESSION_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.writer import EvaluationDataExtractor

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure

from mlrl.util.arrays import enforce_dense
from mlrl.util.options import Options


class RegressionEvaluationDataExtractor(EvaluationDataExtractor):
    """
    Obtains evaluation results according to regression evaluation measures.
    """

    @override
    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any,
                             options: Options):
        ground_truth = enforce_dense(ground_truth, order='C')
        evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, options)

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.values_by_measure(evaluation_measure)[index] = value
