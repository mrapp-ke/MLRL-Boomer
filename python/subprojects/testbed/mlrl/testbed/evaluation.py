"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Dict, Set, Optional

import numpy as np
import sklearn.metrics as metrics
from mlrl.common.arrays import enforce_dense
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.common.options import Options
from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from sklearn.utils.multiclass import is_multilabel

ARGUMENT_HAMMING_LOSS = 'hamming_loss'

ARGUMENT_HAMMING_ACCURACY = 'hamming_accuracy'

ARGUMENT_SUBSET_ZERO_ONE_LOSS = 'subset_zero_one_loss'

ARGUMENT_SUBSET_ACCURACY = 'subset_accuracy'

ARGUMENT_MICRO_PRECISION = 'micro_precision'

ARGUMENT_MICRO_RECALL = 'micro_recall'

ARGUMENT_MICRO_F1 = 'micro_f1'

ARGUMENT_MICRO_JACCARD = 'micro_jaccard'

ARGUMENT_MACRO_PRECISION = 'macro_precision'

ARGUMENT_MACRO_RECALL = 'macro_recall'

ARGUMENT_MACRO_F1 = 'macro_f1'

ARGUMENT_MACRO_JACCARD = 'macro_jaccard'

ARGUMENT_EXAMPLE_WISE_PRECISION = 'example_wise_precision'

ARGUMENT_EXAMPLE_WISE_RECALL = 'example_wise_recall'

ARGUMENT_EXAMPLE_WISE_F1 = 'example_wise_f1'

ARGUMENT_EXAMPLE_WISE_JACCARD = 'example_wise_jaccard'

ARGUMENT_ACCURACY = 'accuracy'

ARGUMENT_ZERO_ONE_LOSS = 'zero_one_loss'

ARGUMENT_PRECISION = 'precision'

ARGUMENT_RECALL = 'recall'

ARGUMENT_F1 = 'f1'

ARGUMENT_JACCARD = 'jaccard'

ARGUMENT_RANK_LOSS = 'rank_loss'

ARGUMENT_COVERAGE_ERROR = 'coverage_error'

ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION = 'lrap'

ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN = 'dcg'

ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 'ndcg'

ARGUMENT_TRAINING_TIME = 'training_time'

ARGUMENT_PREDICTION_TIME = 'prediction_time'


class EvaluationMeasure:

    def __init__(self, argument: str, name: str):
        self.argument = argument
        self.name = name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)


class EvaluationFunction(EvaluationMeasure):

    def __init__(self, argument: str, name: str, evaluation_function, **kwargs):
        super().__init__(argument, name)
        self.evaluation_function = evaluation_function
        self.kwargs = kwargs


ARGS_MICRO = {
    'average': 'micro',
    'zero_division': 1
}

ARGS_MACRO = {
    'average': 'macro',
    'zero_division': 1
}

ARGS_EXAMPLE_WISE = {
    'average': 'samples',
    'zero_division': 1
}

MULTI_LABEL_EVALUATION_FUNCTIONS = [
    EvaluationFunction(ARGUMENT_HAMMING_ACCURACY, 'Hamm. Acc.', lambda a, b: 1 - metrics.hamming_loss(a, b)),
    EvaluationFunction(ARGUMENT_HAMMING_LOSS, 'Hamm. Loss', metrics.hamming_loss),
    EvaluationFunction(ARGUMENT_SUBSET_ACCURACY, 'Subs. Acc.', metrics.accuracy_score),
    EvaluationFunction(ARGUMENT_SUBSET_ZERO_ONE_LOSS, 'Subs. 0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(ARGUMENT_MICRO_PRECISION, 'Mi. Prec.', metrics.precision_score, **ARGS_MICRO),
    EvaluationFunction(ARGUMENT_MICRO_RECALL, 'Mi. Rec.', metrics.recall_score, **ARGS_MICRO),
    EvaluationFunction(ARGUMENT_MICRO_F1, 'Mi. F1', metrics.f1_score, **ARGS_MICRO),
    EvaluationFunction(ARGUMENT_MICRO_JACCARD, 'Mi. Jacc.', metrics.jaccard_score, **ARGS_MICRO),
    EvaluationFunction(ARGUMENT_MACRO_PRECISION, 'Ma. Prec.', metrics.precision_score, **ARGS_MACRO),
    EvaluationFunction(ARGUMENT_MACRO_RECALL, 'Ma. Rec.', metrics.recall_score, **ARGS_MACRO),
    EvaluationFunction(ARGUMENT_MACRO_F1, 'Ma. F1', metrics.f1_score, **ARGS_MACRO),
    EvaluationFunction(ARGUMENT_MACRO_JACCARD, 'Ma. Jacc.', metrics.jaccard_score, **ARGS_MACRO),
    EvaluationFunction(ARGUMENT_EXAMPLE_WISE_PRECISION, 'Ex.-based Prec.', metrics.precision_score,
                       **ARGS_EXAMPLE_WISE),
    EvaluationFunction(ARGUMENT_EXAMPLE_WISE_RECALL, 'Ex.-based Rec.', metrics.recall_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(ARGUMENT_EXAMPLE_WISE_F1, 'Ex.-based F1', metrics.f1_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(ARGUMENT_EXAMPLE_WISE_JACCARD, 'Ex.-based Jacc.', metrics.jaccard_score, **ARGS_EXAMPLE_WISE)
]

SINGLE_LABEL_EVALUATION_FUNCTIONS = [
    EvaluationFunction(ARGUMENT_ACCURACY, 'Acc.', metrics.accuracy_score),
    EvaluationFunction(ARGUMENT_ZERO_ONE_LOSS, '0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(ARGUMENT_PRECISION, 'Prec.', metrics.precision_score),
    EvaluationFunction(ARGUMENT_RECALL, 'Rec.', metrics.recall_score),
    EvaluationFunction(ARGUMENT_F1, 'F1', metrics.f1_score),
    EvaluationFunction(ARGUMENT_JACCARD, 'Jacc.', metrics.jaccard_score)
]

RANKING_EVALUATION_FUNCTIONS = [
    EvaluationFunction(ARGUMENT_RANK_LOSS, 'Rank Loss', metrics.label_ranking_loss),
    EvaluationFunction(ARGUMENT_COVERAGE_ERROR, 'Cov. Error', metrics.coverage_error),
    EvaluationFunction(ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION, 'LRAP', metrics.label_ranking_average_precision_score),
    EvaluationFunction(ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, 'DCG', metrics.dcg_score),
    EvaluationFunction(ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, 'NDCG', metrics.ndcg_score)
]

EVALUATION_MEASURE_TRAINING_TIME = EvaluationMeasure(ARGUMENT_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = EvaluationMeasure(ARGUMENT_PREDICTION_TIME, 'Prediction Time')


class Evaluation(ABC):
    """
    An abstract base class for all classes that evaluate the predictions provided by a classifier or ranker.
    """

    @abstractmethod
    def evaluate(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, predictions, ground_truth,
                 train_time: float, predict_time: float):
        """
        Evaluates the predictions provided by a classifier or ranker.

        :param meta_data:       The meta-data of the data set
        :param data_split:      The split of the available data, the predictions and ground truth labels correspond to
        :param data_type:       Specifies whether the predictions and ground truth labels correspond to the training or
                                test data
        :param predictions:     The predictions provided by the classifier
        :param ground_truth:    The ground truth
        :param train_time:      The time needed to train the model
        :param predict_time:    The time needed to make predictions
        """
        pass


class EvaluationResult:
    """
    Stores the evaluation results according to different measures.
    """

    def __init__(self):
        self.measures: Set[EvaluationMeasure] = set()
        self.results: Optional[List[Dict[EvaluationMeasure, float]]] = None

    def put(self, measure: EvaluationMeasure, score: float, num_folds: int, fold: Optional[int]):
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

    def get(self, measure: EvaluationMeasure, fold: Optional[int]) -> float:
        """
        Returns the score according to a specific measure.

        :param measure: The measure
        :param fold:    The fold, the score corresponds to, or None, if no cross validation is used
        :return:        The score
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        return self.results[fold if fold is not None else 0][measure]

    def dict(self, fold: Optional[int]) -> Dict:
        """
        Returns a dictionary that stores the scores for a specific fold according to each measure.

        :param fold:    The fold, the scores correspond to, or None, if no cross validation is used
        :return:        A dictionary that stores the scores for the given fold according to each measure
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        return self.results[fold if fold is not None else 0].copy()

    def avg(self, measure: EvaluationMeasure) -> (float, float):
        """
        Returns the score and standard deviation according to a specific measure averaged over all available folds.

        :param measure: The measure
        :return:        A tuple consisting of the averaged score and standard deviation
        """
        values = []

        for i in range(len(self.results)):
            if len(self.results[i]) > 0:
                values.append(self.get(measure, i))

        values = np.array(values)
        return np.average(values), np.std(values)

    def avg_dict(self) -> Dict:
        """
        Returns a dictionary that stores the scores, averaged across all folds, as well as the standard deviation,
        according to each measure.

        :return: A dictionary that stores the scores and standard deviation according to each measure
        """
        result: Dict[EvaluationMeasure, float] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure)
            result[measure] = score
            result[EvaluationMeasure(measure.argument, 'Std.-dev. ' + measure.name)] = std_dev.item()

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
    def write_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult, fold: Optional[int]):
        """
        Writes the evaluation results for a single fold to the output.

        :param data_type:           Specifies whether the evaluation results correspond to the training or test data
        :param evaluation_result:   The evaluation result to be written
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used
        """
        pass

    @abstractmethod
    def write_overall_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult,
                                         num_folds: int):
        """
        Writes the overall evaluation results, averaged across all folds, to the output.

        :param data_type:           Specifies whether the evaluation results correspond to the training or test data
        :param evaluation_result:   The evaluation result to be written
        :param num_folds:           The total number of folds
        """
        pass


class EvaluationLogOutput(EvaluationOutput):
    """
    Outputs evaluation result using the logger.
    """

    def __init__(self, options: Options):
        super().__init__(options)

    def write_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult, fold: Optional[int]):
        options = self.options
        text = ''

        for measure in sorted(evaluation_result.measures):
            if measure != EVALUATION_MEASURE_TRAINING_TIME and measure != EVALUATION_MEASURE_PREDICTION_TIME \
                    and options.get_bool(measure.argument, True):
                if len(text) > 0:
                    text += '\n'

                score = evaluation_result.get(measure, fold)
                text += str(measure) + ': ' + str(score)

        log.info('Evaluation result on ' + data_type.value + ' data (Fold ' + str(fold + 1) + '):\n\n%s\n', text)

    def write_overall_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult,
                                         num_folds: int):
        options = self.options
        text = ''

        for measure in sorted(evaluation_result.measures):
            if measure != EVALUATION_MEASURE_TRAINING_TIME and measure != EVALUATION_MEASURE_PREDICTION_TIME \
                    and options.get_bool(measure.argument, True):
                if len(text) > 0:
                    text += '\n'

                score, std_dev = evaluation_result.avg(measure)
                text += (str(measure) + ': ' + str(score))

                if num_folds > 1:
                    text += (' ±' + str(std_dev))

        log.info('Overall evaluation result on ' + data_type.value + ' data:\n\n%s\n', text)


class EvaluationCsvOutput(EvaluationOutput):
    """
    Writes evaluation results to CSV files.
    """

    def __init__(self, options: Options, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        super().__init__(options)
        self.output_dir = output_dir

    def write_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult, fold: Optional[int]):
        options = self.options
        columns = evaluation_result.dict(fold)

        for measure in list(columns.keys()):
            if not options.get_bool(measure.argument, True):
                del columns[measure]

        header = sorted(columns.keys())

        with open_writable_csv_file(self.output_dir, data_type.get_file_name('evaluation'), fold) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)

    def write_overall_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult,
                                         num_folds: int):
        options = self.options
        columns = evaluation_result.avg_dict() if num_folds > 1 else evaluation_result.dict(0)

        for measure in list(columns.keys()):
            if not options.get_bool(measure.argument, True):
                del columns[measure]

        header = sorted(columns.keys())

        with open_writable_csv_file(self.output_dir, data_type.get_file_name('evaluation')) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)


class AbstractEvaluation(Evaluation, ABC):
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

    def evaluate(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, predictions, ground_truth,
                 train_time: float, predict_time: float):
        result = self.results[data_type] if data_type in self.results else EvaluationResult()
        self.results[data_type] = result

        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()
        result.put(EVALUATION_MEASURE_TRAINING_TIME, train_time, num_folds=num_folds, fold=fold)
        result.put(EVALUATION_MEASURE_PREDICTION_TIME, predict_time, num_folds=num_folds, fold=fold)

        self._populate_result(data_split, result, predictions, ground_truth)

        if data_split.is_cross_validation_used():
            for output in self.outputs:
                output.write_evaluation_results(data_type, result, data_split.get_fold())

        if data_split.is_last_fold():
            for output in self.outputs:
                output.write_overall_evaluation_results(data_type, result, data_split.get_num_folds())

    @abstractmethod
    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        pass


class ClassificationEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a single- or multi-label classifier according to commonly used bipartition measures.
    """

    def __init__(self, outputs: List[EvaluationOutput]):
        super(ClassificationEvaluation, self).__init__(outputs)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()

        if is_multilabel(ground_truth):
            evaluation_functions = MULTI_LABEL_EVALUATION_FUNCTIONS
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=DTYPE_UINT8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8))
            evaluation_functions = SINGLE_LABEL_EVALUATION_FUNCTIONS

        for evaluation_function in evaluation_functions:
            if reduce(lambda a, b: a or b.options.get_bool(evaluation_function.argument, True), self.outputs, False):
                kwargs = evaluation_function.kwargs
                score = evaluation_function.evaluation_function(ground_truth, predictions, **kwargs)
                result.put(evaluation_function, score, num_folds=num_folds, fold=fold)


class RankingEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a multi-label ranker according to commonly used ranking measures.
    """

    def __init__(self, outputs: List[EvaluationOutput]):
        super(RankingEvaluation, self).__init__(outputs)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        if is_multilabel(ground_truth):
            num_folds = data_split.get_num_folds()
            fold = data_split.get_fold()
            ground_truth = enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8)

            for evaluation_function in RANKING_EVALUATION_FUNCTIONS:
                if reduce(lambda a, b: a or b.options.get_bool(evaluation_function.argument, True), self.outputs,
                          False):
                    kwargs = evaluation_function.kwargs
                    score = evaluation_function.evaluation_function(ground_truth, predictions, **kwargs)
                    result.put(evaluation_function, score, num_folds=num_folds, fold=fold)
