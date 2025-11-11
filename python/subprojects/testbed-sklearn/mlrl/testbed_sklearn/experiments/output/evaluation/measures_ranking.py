"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement ranking evaluation measures.
"""
from itertools import chain
from typing import Generator

from sklearn import metrics

from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult

from mlrl.testbed.experiments.output.evaluation.measures import Measure


def at_k(measure: Measure) -> Generator[Measure, None, None]:
    """
    Returns the given measure, as well as different variants of it (with varying k) that only take the top-k outputs
    into account.

    :param measure: A measure
    :return:        A generator that iterates the given measure, as well as several variants of it
    """
    yield measure

    for k in [1, 2, 3, 5, 8]:
        yield Measure(
            option_key=measure.option_key,
            name=f'{measure.unformatted_name}@{k}',
            evaluation_function=measure.evaluation_function,
            smaller_is_better=measure.smaller_is_better,
            percentage=measure.percentage,
            **{'k': k},
        )


RANKING_EVALUATION_MEASURES = [
    Measure(
        option_key=EvaluationResult.OPTION_RANK_LOSS,
        name='Ranking Loss',
        smaller_is_better=True,
        evaluation_function=metrics.label_ranking_loss,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_COVERAGE_ERROR,
        name='Coverage Error',
        smaller_is_better=True,
        evaluation_function=metrics.coverage_error,
        percentage=False,
    ),
    Measure(
        option_key=EvaluationResult.OPTION_LABEL_RANKING_AVERAGE_PRECISION,
        name='Label Ranking Average Precision',
        evaluation_function=metrics.label_ranking_average_precision_score,
        percentage=False,
    ),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
] + list(
    chain(
        at_k(
            Measure(
                option_key=EvaluationResult.OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                name='NDCG',
                evaluation_function=metrics.ndcg_score,
            )),
        at_k(
            Measure(
                option_key=EvaluationResult.OPTION_DISCOUNTED_CUMULATIVE_GAIN,
                name='DCG',
                evaluation_function=metrics.dcg_score,
                percentage=False,
            )),
    ))
