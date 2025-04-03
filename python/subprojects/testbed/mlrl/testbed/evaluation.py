"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions provided by a machine learning model according to different measures.
The evaluation results can be written to one or several outputs, e.g., to the console or to a file.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from sklearn import metrics
from sklearn.utils.multiclass import is_multilabel

from mlrl.common.data.arrays import enforce_dense
from mlrl.common.data.types import Float32, Uint8

from mlrl.testbed.experiments.output.data import OutputData, OutputValue
from mlrl.testbed.experiments.output.evaluation.evaluation_result import EVALUATION_MEASURE_PREDICTION_TIME, \
    EVALUATION_MEASURE_TRAINING_TIME, EvaluationResult
from mlrl.testbed.experiments.output.evaluation.measurements import Measurements
from mlrl.testbed.experiments.output.evaluation.measures import Measure
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.fold import Fold


class EvaluationWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that evaluate the predictions provided by a learner and allow to write the
    evaluation results to one or several sinks.
    """

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        fold = state.fold
        fold_index = fold.index if fold.is_cross_validation_used else 0
        sink.write_to_sink(state, output_data, **{EvaluationResult.KWARG_FOLD: fold_index})

        if fold.is_cross_validation_used and fold.is_last_fold:
            overall_fold = replace(fold, index=None, is_last_fold=True)
            sink.write_to_sink(replace(state, fold=overall_fold), output_data)

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        self.results: Dict[str, EvaluationResult] = {}

    @abstractmethod
    def _populate_result(self, fold: Fold, result: EvaluationResult, predictions, ground_truth):
        """
        Must be implemented by subclasses in order to obtain evaluation results and store them in a given
        `EvaluationResult`.

        :param fold:            The fold of the available data that should be used for training and evaluating the model
        :param result:          The `EvaluationResult` that should be used to store the results
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        training_result = state.training_result
        prediction_result = state.prediction_result

        if training_result and prediction_result:
            dataset = state.dataset
            data_type = dataset.type
            result = self.results[data_type] if data_type in self.results else EvaluationResult()
            self.results[data_type] = result
            fold = state.fold
            result.put(EVALUATION_MEASURE_TRAINING_TIME,
                       training_result.training_duration.value,
                       num_folds=fold.num_folds,
                       fold=fold.index)
            result.put(EVALUATION_MEASURE_PREDICTION_TIME,
                       prediction_result.prediction_duration.value,
                       num_folds=fold.num_folds,
                       fold=fold.index)
            self._populate_result(fold, result, prediction_result.predictions, dataset.y)
            return result
        return None


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


class BinaryEvaluationWriter(EvaluationWriter):
    """
    Evaluates the quality of binary predictions provided by a single- or multi-label classifier according to commonly
    used bipartition measures.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.multi_label_evaluation_measures = OutputValue.filter_values(MULTI_LABEL_EVALUATION_MEASURES, *options)
        self.single_label_evaluation_measures = OutputValue.filter_values(SINGLE_LABEL_EVALUATION_MEASURES, *options)

    def _populate_result(self, fold: Fold, result: EvaluationResult, predictions, ground_truth):
        if is_multilabel(ground_truth):
            evaluation_measures = self.multi_label_evaluation_measures
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=Uint8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=Uint8))
            evaluation_measures = self.single_label_evaluation_measures

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                score = evaluation_measure.evaluate(ground_truth, predictions)
                result.put(evaluation_measure, score, num_folds=fold.num_folds, fold=fold.index)


class RegressionEvaluationWriter(EvaluationWriter):
    """
    Evaluates the quality of scores provided by a single- or multi-output regressor according to commonly used
    regression measures.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.regression_evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, *options)

    def _populate_result(self, fold: Fold, result: EvaluationResult, predictions, ground_truth):
        ground_truth = enforce_dense(ground_truth, order='C', dtype=Float32)
        evaluation_measures = self.regression_evaluation_measures

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                score = evaluation_measure.evaluate(ground_truth, predictions)
                result.put(evaluation_measure, score, num_folds=fold.num_folds, fold=fold.index)


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


class RankingEvaluationWriter(EvaluationWriter):
    """
    Evaluates the quality of scores provided by a single- or multi-label classifier according to commonly used
    regression and ranking measures.
    """

    def __init__(self, *sinks: Sink):
        super().__init__(*sinks)
        options = [sink.options for sink in sinks]
        self.regression_evaluation_measures = OutputValue.filter_values(REGRESSION_EVALUATION_MEASURES, *options)
        self.ranking_evaluation_measures = OutputValue.filter_values(RANKING_EVALUATION_MEASURES, *options)

    def _populate_result(self, fold: Fold, result: EvaluationResult, predictions, ground_truth):
        ground_truth = enforce_dense(ground_truth, order='C', dtype=Uint8)

        if is_multilabel(ground_truth):
            evaluation_measures = self.ranking_evaluation_measures + self.regression_evaluation_measures
        else:
            evaluation_measures = self.regression_evaluation_measures

            if predictions.shape[1] > 1:
                predictions = predictions[:, -1]

        for evaluation_measure in evaluation_measures:
            if isinstance(evaluation_measure, Measure):
                score = evaluation_measure.evaluate(ground_truth, predictions)
                result.put(evaluation_measure, score, num_folds=fold.num_folds, fold=fold.index)
