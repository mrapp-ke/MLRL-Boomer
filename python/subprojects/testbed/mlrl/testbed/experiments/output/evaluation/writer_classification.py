"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to classification evaluation measures to one or several
sinks.
"""

from typing import Any

import numpy as np

from sklearn.utils.multiclass import is_multilabel

from mlrl.common.data.arrays import enforce_dense
from mlrl.common.data.types import Uint8

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.evaluation.measures_classification import MULTI_LABEL_EVALUATION_MEASURES, \
    SINGLE_LABEL_EVALUATION_MEASURES
from mlrl.testbed.experiments.output.evaluation.writer import EvaluationWriter
from mlrl.testbed.experiments.output.sinks import Sink


class ClassificationEvaluationWriter(EvaluationWriter):
    """
    Allows writing evaluation results according to classification evaluation measures to one or several sinks.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.multi_label_evaluation_measures = OutputValue.filter_values(MULTI_LABEL_EVALUATION_MEASURES, *options)
        self.single_label_evaluation_measures = OutputValue.filter_values(SINGLE_LABEL_EVALUATION_MEASURES, *options)

    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any):
        if is_multilabel(ground_truth):
            evaluation_measures = self.multi_label_evaluation_measures
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=Uint8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=Uint8))
            evaluation_measures = self.single_label_evaluation_measures

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.values_by_measure(evaluation_measure)[index] = value
