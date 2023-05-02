"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod

import numpy as np
import sklearn.metrics as metrics
from mlrl.common.arrays import enforce_dense
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.common.options import Options
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import Formattable, filter_formattables, format_table, OPTION_DECIMALS, OPTION_PERCENTAGE
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.predictions import PredictionScope
from sklearn.utils.multiclass import is_multilabel
from typing import List, Dict, Set, Optional, Tuple

OPTION_ENABLE_ALL = 'enable_all'

OPTION_HAMMING_LOSS = 'hamming_loss'

OPTION_HAMMING_ACCURACY = 'hamming_accuracy'

OPTION_SUBSET_ZERO_ONE_LOSS = 'subset_zero_one_loss'

OPTION_SUBSET_ACCURACY = 'subset_accuracy'

OPTION_MICRO_PRECISION = 'micro_precision'

OPTION_MICRO_RECALL = 'micro_recall'

OPTION_MICRO_F1 = 'micro_f1'

OPTION_MICRO_JACCARD = 'micro_jaccard'

OPTION_MACRO_PRECISION = 'macro_precision'

OPTION_MACRO_RECALL = 'macro_recall'

OPTION_MACRO_F1 = 'macro_f1'

OPTION_MACRO_JACCARD = 'macro_jaccard'

OPTION_EXAMPLE_WISE_PRECISION = 'example_wise_precision'

OPTION_EXAMPLE_WISE_RECALL = 'example_wise_recall'

OPTION_EXAMPLE_WISE_F1 = 'example_wise_f1'

OPTION_EXAMPLE_WISE_JACCARD = 'example_wise_jaccard'

OPTION_ACCURACY = 'accuracy'

OPTION_ZERO_ONE_LOSS = 'zero_one_loss'

OPTION_PRECISION = 'precision'

OPTION_RECALL = 'recall'

OPTION_F1 = 'f1'

OPTION_JACCARD = 'jaccard'

OPTION_MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'

OPTION_MEAN_SQUARED_ERROR = 'mean_squared_error'

OPTION_MEDIAN_ABSOLUTE_ERROR = 'mean_absolute_error'

OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'

OPTION_RANK_LOSS = 'rank_loss'

OPTION_COVERAGE_ERROR = 'coverage_error'

OPTION_LABEL_RANKING_AVERAGE_PRECISION = 'lrap'

OPTION_DISCOUNTED_CUMULATIVE_GAIN = 'dcg'

OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 'ndcg'

OPTION_TRAINING_TIME = 'training_time'

OPTION_PREDICTION_TIME = 'prediction_time'


class EvaluationFunction(Formattable):
    """
    An evaluation function.
    """

    def __init__(self, option: str, name: str, evaluation_function, percentage: bool = True, **kwargs):
        """
        :param evaluation_function: The function that should be invoked for evaluation
        """
        super().__init__(option, name, percentage)
        self.evaluation_function = evaluation_function
        self.kwargs = kwargs

    def evaluate(self, ground_truth, predictions) -> float:
        """
        Applies the evaluation function to given predictions and ground truth labels.

        :param ground_truth:    The ground truth
        :param predictions:     The predictions
        :return:                An evaluation score
        """
        return self.evaluation_function(ground_truth, predictions, **self.kwargs)


ARGS_SINGLE_LABEL = {'zero_division': 1}

ARGS_MICRO = {'average': 'micro', 'zero_division': 1}

ARGS_MACRO = {'average': 'macro', 'zero_division': 1}

ARGS_EXAMPLE_WISE = {'average': 'samples', 'zero_division': 1}

EVALUATION_MEASURE_TRAINING_TIME = Formattable(OPTION_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = Formattable(OPTION_PREDICTION_TIME, 'Prediction Time')

MULTI_LABEL_EVALUATION_MEASURES: List[Formattable] = [
    EvaluationFunction(OPTION_HAMMING_ACCURACY, 'Hamming Accuracy', lambda a, b: 1 - metrics.hamming_loss(a, b)),
    EvaluationFunction(OPTION_HAMMING_LOSS, 'Hamming Loss', metrics.hamming_loss),
    EvaluationFunction(OPTION_SUBSET_ACCURACY, 'Subset Accuracy', metrics.accuracy_score),
    EvaluationFunction(OPTION_SUBSET_ZERO_ONE_LOSS, 'Subset 0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(OPTION_MICRO_PRECISION, 'Micro Precision', metrics.precision_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_RECALL, 'Micro Recall', metrics.recall_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_F1, 'Micro F1', metrics.f1_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_JACCARD, 'Micro Jaccard', metrics.jaccard_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MACRO_PRECISION, 'Macro Precision', metrics.precision_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_RECALL, 'Macro Recall', metrics.recall_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_F1, 'Macro F1', metrics.f1_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_JACCARD, 'Macro Jaccard', metrics.jaccard_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_EXAMPLE_WISE_PRECISION, 'Example-wise Precision', metrics.precision_score,
                       **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_RECALL, 'Example-wise Recall', metrics.recall_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_F1, 'Example-wise F1', metrics.f1_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_JACCARD, 'Example-wise Jaccard', metrics.jaccard_score, **ARGS_EXAMPLE_WISE),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

SINGLE_LABEL_EVALUATION_MEASURES: List[Formattable] = [
    EvaluationFunction(OPTION_ACCURACY, 'Accuracy', metrics.accuracy_score),
    EvaluationFunction(OPTION_ZERO_ONE_LOSS, '0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(OPTION_PRECISION, 'Precision', metrics.precision_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_RECALL, 'Recall', metrics.recall_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_F1, 'F1', metrics.f1_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_JACCARD, 'Jaccard', metrics.jaccard_score, **ARGS_SINGLE_LABEL),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

REGRESSION_EVALUATION_MEASURES: List[Formattable] = [
    EvaluationFunction(OPTION_MEAN_ABSOLUTE_ERROR, 'Mean Absolute Error', metrics.mean_absolute_error,
                       percentage=False),
    EvaluationFunction(OPTION_MEAN_SQUARED_ERROR, 'Mean Squared Error', metrics.mean_squared_error, percentage=False),
    EvaluationFunction(OPTION_MEDIAN_ABSOLUTE_ERROR,
                       'Median Absolute Error',
                       metrics.median_absolute_error,
                       percentage=False),
    EvaluationFunction(OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                       'Mean Absolute Percentage Error',
                       metrics.mean_absolute_percentage_error,
                       percentage=False),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

RANKING_EVALUATION_MEASURES: List[Formattable] = [
    EvaluationFunction(OPTION_RANK_LOSS, 'Ranking Loss', metrics.label_ranking_loss, percentage=False),
    EvaluationFunction(OPTION_COVERAGE_ERROR, 'Coverage Error', metrics.coverage_error, percentage=False),
    EvaluationFunction(OPTION_LABEL_RANKING_AVERAGE_PRECISION,
                       'Label Ranking Average Precision',
                       metrics.label_ranking_average_precision_score,
                       percentage=False),
    EvaluationFunction(OPTION_DISCOUNTED_CUMULATIVE_GAIN,
                       'Discounted Cumulative Gain',
                       metrics.dcg_score,
                       percentage=False),
    EvaluationFunction(OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, 'NDCG', metrics.ndcg_score),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]


class EvaluationResult:
    """
    Stores the evaluation results according to different measures.
    """

    def __init__(self):
        self.measures: Set[Formattable] = set()
        self.results: Optional[List[Dict[Formattable, float]]] = None

    def put(self, measure: Formattable, score: float, num_folds: int, fold: Optional[int]):
        """
        Adds a new score according to a specific measure to the evaluation result.

        :param measure:     The measure
        :param score:       The score according to the measure
        :param num_folds:   The total number of cross validation folds
        :param fold:        The fold, the score corresponds to, or None, if no cross validation is used
        """
        if self.results is None:
            self.results = [{} for _ in range(num_folds)]
        elif len(self.results) != num_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(measure)
        values = self.results[fold if fold is not None else 0]
        values[measure] = score

    def get(self, measure: Formattable, fold: Optional[int], **kwargs) -> str:
        """
        Returns the score according to a specific measure.

        :param measure: The measure
        :param fold:    The fold, the score corresponds to, or None, if no cross validation is used
        :return:        A textual representation of the score
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        score = self.results[fold if fold is not None else 0][measure]
        return measure.format(score, **kwargs)

    def dict(self, fold: Optional[int], **kwargs) -> Dict[Formattable, str]:
        """
        Returns a dictionary that stores the scores for a specific fold according to each measure.

        :param fold:    The fold, the scores correspond to, or None, if no cross validation is used
        :return:        A dictionary that stores textual representations of the scores for the given fold according to
                        each measure
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        results: Dict[Formattable, str] = {}

        for measure, score in self.results[fold if fold is not None else 0].items():
            results[measure] = measure.format(score, **kwargs)

        return results

    def avg(self, measure: Formattable, **kwargs) -> Tuple[str, str]:
        """
        Returns the score and standard deviation according to a specific measure averaged over all available folds.

        :param measure: The measure
        :return:        A tuple consisting of textual representations of the averaged score and standard deviation
        """
        values = []

        for i in range(len(self.results)):
            results = self.results[i]

            if len(results) > 0:
                values.append(results[measure])

        values = np.array(values)
        return measure.format(np.average(values), **kwargs), measure.format(np.std(values), **kwargs)

    def avg_dict(self, **kwargs) -> Dict[Formattable, str]:
        """
        Returns a dictionary that stores the scores, averaged across all folds, as well as the standard deviation,
        according to each measure.

        :return: A dictionary that stores textual representations of the scores and standard deviation according to each
                 measure
        """
        result: Dict[Formattable, str] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure, **kwargs)
            result[measure] = score
            result[Formattable(measure.option, 'Std.-dev. ' + measure.name, measure.percentage)] = std_dev

        return result


class EvaluationOutput(ABC):
    """
    An abstract base class for all outputs, evaluation results may be written to.
    """

    def __init__(self, options: Options):
        """
        :param options: The options that should be used for writing evaluation results to the output
        """
        self.options = options

    @abstractmethod
    def write_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                 evaluation_result: EvaluationResult, fold: Optional[int]):
        """
        Writes the evaluation results for a single fold to the output.

        :param data_type:           Specifies whether the evaluation results correspond to the training or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param evaluation_result:   The evaluation result to be written
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used
        """
        pass

    @abstractmethod
    def write_overall_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                         evaluation_result: EvaluationResult, num_folds: int):
        """
        Writes the overall evaluation results, averaged across all folds, to the output.

        :param data_type:           Specifies whether the evaluation results correspond to the training or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param evaluation_result:   The evaluation result to be written
        :param num_folds:           The total number of folds
        """
        pass


class EvaluationLogOutput(EvaluationOutput):
    """
    Outputs evaluation results using the logger.
    """

    def __init__(self, options: Options):
        super().__init__(options)
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 2)

    def write_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                 evaluation_result: EvaluationResult, fold: Optional[int]):
        options = self.options
        rows = []
        enable_all = options.get_bool(OPTION_ENABLE_ALL, True)

        for measure in sorted(evaluation_result.measures):
            if options.get_bool(measure.option, enable_all) and measure != EVALUATION_MEASURE_TRAINING_TIME \
                    and measure != EVALUATION_MEASURE_PREDICTION_TIME:
                score = evaluation_result.get(measure, fold, percentage=self.percentage, decimals=self.decimals)
                rows.append([str(measure), score])

        model_size = '' if prediction_scope.is_global() else 'using a model of size ' + str(
            prediction_scope.get_model_size()) + ' '
        log.info('Evaluation result on %s data %s(Fold %s):\n\n%s\n', data_type.value, model_size, str(fold + 1),
                 format_table(rows))

    def write_overall_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                         evaluation_result: EvaluationResult, num_folds: int):
        options = self.options
        rows = []
        enable_all = options.get_bool(OPTION_ENABLE_ALL, True)

        for measure in sorted(evaluation_result.measures):
            if options.get_bool(measure.option, enable_all) and measure != EVALUATION_MEASURE_TRAINING_TIME \
                    and measure != EVALUATION_MEASURE_PREDICTION_TIME:
                score, std_dev = evaluation_result.avg(measure, percentage=self.percentage, decimals=self.decimals)
                row = [str(measure), score]

                if num_folds > 1:
                    row.append('±' + std_dev)

                rows.append(row)

        model_size = '' if prediction_scope.is_global() else ' using a model of size ' + str(
            prediction_scope.get_model_size())
        log.info('Overall evaluation result on %s data%s:\n\n%s\n', data_type.value, model_size, format_table(rows))


class EvaluationCsvOutput(EvaluationOutput):
    """
    Writes evaluation results to CSV files.
    """

    COLUMN_MODEL_SIZE = 'Model size'

    def __init__(self, options: Options, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        super().__init__(options)
        self.output_dir = output_dir
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 0)

    def write_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                 evaluation_result: EvaluationResult, fold: Optional[int]):
        columns: Dict = evaluation_result.dict(fold, percentage=self.percentage, decimals=self.decimals)
        header = list(columns.keys())
        options = self.options
        enable_all = options.get_bool(OPTION_ENABLE_ALL, True)

        for formattable in header:
            if not options.get_bool(formattable.option, enable_all):
                del columns[formattable]

        header = sorted(columns.keys())
        incremental_prediction = not prediction_scope.is_global()

        if incremental_prediction:
            columns[EvaluationCsvOutput.COLUMN_MODEL_SIZE] = prediction_scope.get_model_size()
            header = [EvaluationCsvOutput.COLUMN_MODEL_SIZE] + header

        with open_writable_csv_file(self.output_dir,
                                    data_type.get_file_name('evaluation'),
                                    fold,
                                    append=incremental_prediction) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)

    def write_overall_evaluation_results(self, data_type: DataType, prediction_scope: PredictionScope,
                                         evaluation_result: EvaluationResult, num_folds: int):
        columns: Dict = evaluation_result.avg_dict(percentage=self.percentage, decimals=self.decimals) \
            if num_folds > 1 else evaluation_result.dict(0, percentage=self.percentage, decimals=self.decimals)
        header = list(columns.keys())
        options = self.options
        enable_all = options.get_bool(OPTION_ENABLE_ALL, True)

        for formattable in header:
            if not options.get_bool(formattable.option, enable_all):
                del columns[formattable]

        header = sorted(columns.keys())
        incremental_prediction = not prediction_scope.is_global()

        if incremental_prediction:
            columns[EvaluationCsvOutput.COLUMN_MODEL_SIZE] = prediction_scope.get_model_size()
            header = [EvaluationCsvOutput.COLUMN_MODEL_SIZE] + header

        with open_writable_csv_file(self.output_dir,
                                    data_type.get_file_name('evaluation'),
                                    append=incremental_prediction) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)


class EvaluationPrinter(ABC):
    """
    An abstract base class for all classes that evaluate the predictions provided by a classifier or ranker and allow to
    write the results to one or several outputs.
    """

    def __init__(self, outputs: List[EvaluationOutput]):
        """
        :param outputs: The outputs, the evaluation results should be written to
        """
        self.outputs = outputs
        self.results: Dict[str, EvaluationResult] = {}

    def evaluate(self, data_split: DataSplit, data_type: DataType, prediction_scope: PredictionScope, predictions,
                 ground_truth, train_time: float, predict_time: float):
        """
        Evaluates the predictions provided by a classifier or ranker and prints the evaluation results.

        :param data_split:          The split of the available data, the predictions and ground truth labels correspond
                                    to
        :param data_type:           Specifies whether the predictions and ground truth labels correspond to the training
                                    or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param predictions:         The predictions provided by the classifier
        :param ground_truth:        The ground truth
        :param train_time:          The time needed to train the model
        :param predict_time:        The time needed to make predictions
        """
        result = self.results[data_type] if data_type in self.results else EvaluationResult()
        self.results[data_type] = result

        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()
        result.put(EVALUATION_MEASURE_TRAINING_TIME, train_time, num_folds=num_folds, fold=fold)
        result.put(EVALUATION_MEASURE_PREDICTION_TIME, predict_time, num_folds=num_folds, fold=fold)

        self._populate_result(data_split, result, predictions, ground_truth)

        if data_split.is_cross_validation_used():
            for output in self.outputs:
                output.write_evaluation_results(data_type, prediction_scope, result, data_split.get_fold())

        if data_split.is_last_fold():
            for output in self.outputs:
                output.write_overall_evaluation_results(data_type, prediction_scope, result, data_split.get_num_folds())

    @abstractmethod
    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        pass


class BinaryEvaluationPrinter(EvaluationPrinter):
    """
    Evaluates the quality of binary predictions provided by a single- or multi-label classifier according to commonly
    used bipartition measures.
    """

    def __init__(self, outputs: List[EvaluationOutput]):
        super(BinaryEvaluationPrinter, self).__init__(outputs)
        options = [output.options for output in outputs]
        self.multi_Label_evaluation_functions = filter_formattables(MULTI_LABEL_EVALUATION_MEASURES, options)
        self.single_Label_evaluation_functions = filter_formattables(SINGLE_LABEL_EVALUATION_MEASURES, options)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()

        if is_multilabel(ground_truth):
            evaluation_functions = self.multi_Label_evaluation_functions
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=DTYPE_UINT8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8))
            evaluation_functions = self.single_Label_evaluation_functions

        for evaluation_function in evaluation_functions:
            if isinstance(evaluation_function, EvaluationFunction):
                score = evaluation_function.evaluate(ground_truth, predictions)
                result.put(evaluation_function, score, num_folds=num_folds, fold=fold)


class ScoreEvaluationPrinter(EvaluationPrinter):
    """
    Evaluates the quality of regression scores provided by a single- or multi-output regressor according to commonly
    used regression and ranking measures.
    """

    def __init__(self, outputs: List[EvaluationOutput]):
        super(ScoreEvaluationPrinter, self).__init__(outputs)
        options = [output.options for output in outputs]
        self.regression_evaluation_functions = filter_formattables(REGRESSION_EVALUATION_MEASURES, options)
        self.ranking_evaluation_functions = filter_formattables(RANKING_EVALUATION_MEASURES, options)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()
        ground_truth = enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8)

        if is_multilabel(ground_truth):
            evaluation_functions = self.ranking_evaluation_functions + self.regression_evaluation_functions
        else:
            evaluation_functions = self.regression_evaluation_functions

            if predictions.shape[1] > 1:
                predictions = predictions[:, -1]

        for evaluation_function in evaluation_functions:
            if isinstance(evaluation_function, EvaluationFunction):
                score = evaluation_function.evaluate(ground_truth, predictions)
                result.put(evaluation_function, score, num_folds=num_folds, fold=fold)


class ProbabilityEvaluationPrinter(ScoreEvaluationPrinter):
    """
    Evaluates the quality of probability estimates provided by a single- or multi-label classifier according to commonly
    used regression and ranking measures.
    """
    pass
