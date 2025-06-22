"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement ranking evaluation measures.
"""
from sklearn import metrics

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult

from mlrl.testbed.experiments.output.evaluation.measures import Measure

RANKING_EVALUATION_MEASURES = [
    Measure(
        option_key=EvaluationResult.OPTION_RANK_LOSS,
        name='Ranking Loss',
        evaluation_function=metrics.label_ranking_loss,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_COVERAGE_ERROR,
        name='Coverage Error',
        evaluation_function=metrics.coverage_error,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
        name='Label Ranking Average Precision',
        evaluation_function=metrics.label_ranking_average_precision_score,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
        name='Discounted Cumulative Gain',
        evaluation_function=metrics.dcg_score,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
        name='NDCG',
        evaluation_function=metrics.ndcg_score,
    ),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]
