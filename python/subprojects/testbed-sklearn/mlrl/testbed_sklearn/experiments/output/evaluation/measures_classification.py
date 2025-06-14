"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement classification evaluation measures.
"""
from sklearn import metrics

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult

from mlrl.testbed.experiments.output.evaluation.measures import Measure

ARGS_SINGLE_LABEL = {'zero_division': 1}

ARGS_MICRO = {'average': 'micro', 'zero_division': 1}

ARGS_MACRO = {'average': 'macro', 'zero_division': 1}

ARGS_EXAMPLE_WISE = {'average': 'samples', 'zero_division': 1}

MULTI_LABEL_EVALUATION_MEASURES = [
    Measure(
        option_key=EvaluationResult.OPTION_HAMMING_ACCURACY,
        name='Hamming Accuracy',
        evaluation_function=lambda a, b: 1 - metrics.hamming_loss(a, b),
    ),
    Measure(
        option_key=EvaluationResult.OPTION_HAMMING_LOSS,
        name='Hamming Loss',
        evaluation_function=metrics.hamming_loss,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_SUBSET_ACCURACY,
        name='Subset Accuracy',
        evaluation_function=metrics.accuracy_score,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_SUBSET_ZERO_ONE_LOSS,
        name='Subset 0/1 Loss',
        evaluation_function=lambda a, b: 1 - metrics.accuracy_score(a, b),
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MICRO_PRECISION,
        name='Micro Precision',
        evaluation_function=metrics.precision_score,
        **ARGS_MICRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MICRO_RECALL,
        name='Micro Recall',
        evaluation_function=metrics.recall_score,
        **ARGS_MICRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MICRO_F1,
        name='Micro F1',
        evaluation_function=metrics.f1_score,
        **ARGS_MICRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MICRO_JACCARD,
        name='Micro Jaccard',
        evaluation_function=metrics.jaccard_score,
        **ARGS_MICRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MACRO_PRECISION,
        name='Macro Precision',
        evaluation_function=metrics.precision_score,
        **ARGS_MACRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MACRO_RECALL,
        name='Macro Recall',
        evaluation_function=metrics.recall_score,
        **ARGS_MACRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MACRO_F1,
        name='Macro F1',
        evaluation_function=metrics.f1_score,
        **ARGS_MACRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_MACRO_JACCARD,
        name='Macro Jaccard',
        evaluation_function=metrics.jaccard_score,
        **ARGS_MACRO,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_EXAMPLE_WISE_PRECISION,
        name='Example-wise Precision',
        evaluation_function=metrics.precision_score,
        **ARGS_EXAMPLE_WISE,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_EXAMPLE_WISE_RECALL,
        name='Example-wise Recall',
        evaluation_function=metrics.recall_score,
        **ARGS_EXAMPLE_WISE,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_EXAMPLE_WISE_F1,
        name='Example-wise F1',
        evaluation_function=metrics.f1_score,
        **ARGS_EXAMPLE_WISE,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_EXAMPLE_WISE_JACCARD,
        name='Example-wise Jaccard',
        evaluation_function=metrics.jaccard_score,
        **ARGS_EXAMPLE_WISE,
    ),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

SINGLE_LABEL_EVALUATION_MEASURES = [
    Measure(
        option_key=EvaluationResult.OPTION_ACCURACY,
        name='Accuracy',
        evaluation_function=metrics.accuracy_score,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_ZERO_ONE_LOSS,
        name='0/1 Loss',
        evaluation_function=lambda a, b: 1 - metrics.accuracy_score(a, b),
    ),
    Measure(
        option_key=EvaluationResult.OPTION_PRECISION,
        name='Precision',
        evaluation_function=metrics.precision_score,
        **ARGS_SINGLE_LABEL,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_RECALL,
        name='Recall',
        evaluation_function=metrics.recall_score,
        **ARGS_SINGLE_LABEL,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_F1,
        name='F1',
        evaluation_function=metrics.f1_score,
        **ARGS_SINGLE_LABEL,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_JACCARD,
        name='Jaccard',
        evaluation_function=metrics.jaccard_score,
        **ARGS_SINGLE_LABEL,
    ),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]
