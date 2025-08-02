"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results according to classification evaluation measures to one or several
sinks.
"""

from typing import Any, override

import numpy as np

from sklearn.utils.multiclass import is_multilabel

from mlrl.testbed_sklearn.experiments.output.evaluation.measures_classification import \
    MULTI_LABEL_EVALUATION_MEASURES, SINGLE_LABEL_EVALUATION_MEASURES
from mlrl.testbed_sklearn.experiments.output.evaluation.writer import EvaluationDataExtractor

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure

from mlrl.util.arrays import enforce_dense
from mlrl.util.options import Options


class ClassificationEvaluationDataExtractor(EvaluationDataExtractor):
    """
    Obtains evaluation results according to classification evaluation measures.
    """

    @override
    def _update_measurements(self, measurements: Measurements, index: int, ground_truth: Any, predictions: Any,
                             options: Options):
        if is_multilabel(ground_truth):
            evaluation_measures = OutputValue.filter_values(MULTI_LABEL_EVALUATION_MEASURES, options)
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C'))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C'))
            evaluation_measures = OutputValue.filter_values(SINGLE_LABEL_EVALUATION_MEASURES, options)

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                value = evaluation_measure.evaluate(ground_truth, predictions)
                measurements.values_by_measure(evaluation_measure)[index] = value
