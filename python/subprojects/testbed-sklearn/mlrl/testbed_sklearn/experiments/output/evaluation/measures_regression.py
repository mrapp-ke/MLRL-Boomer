"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement regression evaluation measures.
"""
from sklearn import metrics

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult

from mlrl.testbed.experiments.output.evaluation.measures import Measure

REGRESSION_EVALUATION_MEASURES = [
    Measure(
        option_key=EvaluationResult.OPTION_MEAN_ABSOLUTE_ERROR,
        name='Mean Absolute Error',
        evaluation_function=metrics.mean_absolute_error,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MEAN_SQUARED_ERROR,
        name='Mean Squared Error',
        evaluation_function=metrics.mean_squared_error,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MEDIAN_ABSOLUTE_ERROR,
        name='Median Absolute Error',
        evaluation_function=metrics.median_absolute_error,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
        name='Mean Absolute Percentage Error',
        evaluation_function=metrics.mean_absolute_percentage_error,
        percentage=False,
    ),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]
