"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional

import numpy as np
import sklearn.metrics as metrics
from mlrl.common.arrays import enforce_dense
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.testbed.data import MetaData
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.training import DataSplit, DataType
from sklearn.utils.multiclass import is_multilabel

# The name of the accuracy metric
ACCURACY = 'Acc.'

# The name of the 0/1 loss metric.
ZERO_ONE_LOSS = '0/1 Loss'

# The name of the precision metric
PRECISION = 'Prec.'

# The name of the recall metric
RECALL = 'Rec.'

# The name of the F1 metric
F1 = 'F1'

# The name of the Jaccard metric
JACCARD = 'Jacc.'

# The name of the hamming loss metric
HAMMING_LOSS = 'Hamm. Loss'

# The name of the hamming accuracy metric
HAMMING_ACCURACY = 'Hamm. Acc.'

# The name of the subset 0/1 loss metric
SUBSET_ZERO_ONE_LOSS = 'Subs. 0/1 Loss'

# The name of the subset accuracy metric
SUBSET_ACCURACY = 'Subs. Acc.'

# The name of the micro-averaged precision metric
MICRO_PRECISION = 'Mi. Prec.'

# The name of the macro-averaged precision metric
MACRO_PRECISION = 'Ma. Prec.'

# The name of the example-based precision metric
EX_BASED_PRECISION = 'Ex.-based Prec.'

# The name of the micro-averaged recall metric
MICRO_RECALL = 'Mi. Rec.'

# The name of the macro-averaged recall metric
MACRO_RECALL = 'Ma. Rec.'

# The name of the example-based recall metric
EX_BASED_RECALL = 'Ex.-based Rec.'

# The name of the micro-averaged F1 metric
MICRO_F1 = 'Mi. F1'

# The name of the macro-averaged F1 metric
MACRO_F1 = 'Ma. F1'

# The name of the example-based F1 metric
EX_BASED_F1 = 'Ex.-based F1'

# The name of the micro-averaged Jaccard metric
MICRO_JACCARD = 'Mi. Jacc.'

# The name of the macro-averaged Jaccard metric
MACRO_JACCARD = 'Ma. Jacc.'

# The name of the example-based Jaccard metric
EX_BASED_JACCARD = 'Ex.-based Jacc.'

# The name of the rank loss metric
RANK_LOSS = 'Rank Loss'

# The name of the coverage error metric
COVERAGE_ERROR = 'Cov. Error'

# The name of the label ranking average precision metric
LABEL_RANKING_AVERAGE_PRECISION = 'LRAP'

# The name of the discounted cumulative gain metric
DISCOUNTED_CUMULATIVE_GAIN = 'DCG'

# The name of the normalized discounted cumulative gain metric
NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 'NDCG'

# The time needed to train the model
TIME_TRAIN = 'Training Time'

# The time needed to make predictions
TIME_PREDICT = 'Prediction Time'


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
        self.measures: Set[str] = set()
        self.results: Optional[List[Dict[str, float]]] = None

    def put(self, name: str, score: float, num_folds: int, fold: Optional[int]):
        """
        Adds a new score according to a specific measure to the evaluation result.

        :param name:        The name of the measure
        :param score:       The score according to the measure
        :param num_folds:   The total number of cross validation folds
        :param fold:        The fold, the score corresponds to, or None, if no cross validation is used
        """
        if self.results is None:
            self.results = [{} for _ in range(num_folds)]
        elif len(self.results) != num_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(name)
        values = self.results[fold if fold is not None else 0]
        values[name] = score

    def get(self, name: str, fold: Optional[int]) -> float:
        """
        Returns the score according to a specific measure.

        :param name:    The name of the measure
        :param fold:    The fold, the score corresponds to, or None, if no cross validation is used
        :return:        The score
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        return self.results[fold if fold is not None else 0][name]

    def dict(self, fold: Optional[int]) -> Dict:
        """
        Returns a dictionary that stores the scores for a specific fold according to each measure.

        :param fold:    The fold, the scores correspond to, or None, if no cross validation is used
        :return:        A dictionary that stores the scores for the given fold according to each measure
        """
        if self.results is None:
            raise AssertionError('No evaluation results available')

        return self.results[fold if fold is not None else 0].copy()

    def avg(self, name: str) -> (float, float):
        """
        Returns the score and standard deviation according to a specific measure averaged over all available folds.

        :param name:    The name of the measure
        :return:        A tuple consisting of the averaged score and standard deviation
        """
        values = []

        for i in range(len(self.results)):
            if len(self.results[i]) > 0:
                values.append(self.get(name, i))

        values = np.array(values)
        return np.average(values), np.std(values)

    def avg_dict(self) -> Dict:
        """
        Returns a dictionary that stores the scores, averaged across all folds, as well as the standard deviation,
        according to each measure.

        :return: A dictionary that stores the scores and standard deviation according to each measure
        """
        result: Dict[str, float] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure)
            result[measure] = score
            result['Std.-dev. ' + measure] = std_dev.item()

        return result


class EvaluationOutput(ABC):
    """
    An abstract base class for all outputs, evaluation results may be written to.
    """

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

    def write_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult, fold: Optional[int]):
        text = ''

        for measure in sorted(evaluation_result.measures):
            if measure != TIME_TRAIN and measure != TIME_PREDICT:
                if len(text) > 0:
                    text += '\n'

                score = evaluation_result.get(measure, fold)
                text += measure + ': ' + str(score)

        log.info('Evaluation result on ' + data_type.value + ' data (Fold ' + str(fold + 1) + '):\n\n%s\n', text)

    def write_overall_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult,
                                         num_folds: int):
        text = ''

        for measure in sorted(evaluation_result.measures):
            if measure != TIME_TRAIN and measure != TIME_PREDICT:
                if len(text) > 0:
                    text += '\n'

                score, std_dev = evaluation_result.avg(measure)
                text += (measure + ': ' + str(score))

                if num_folds > 1:
                    text += (' ±' + str(std_dev))

        log.info('Overall evaluation result on ' + data_type.value + ' data:\n\n%s\n', text)


class EvaluationCsvOutput(EvaluationOutput):
    """
    Writes evaluation results to CSV files.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult, fold: Optional[int]):
        columns = evaluation_result.dict(fold)
        header = sorted(columns.keys())

        with open_writable_csv_file(self.output_dir, 'evaluation_' + data_type.value, fold) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)

    def write_overall_evaluation_results(self, data_type: DataType, evaluation_result: EvaluationResult,
                                         num_folds: int):
        columns = evaluation_result.avg_dict() if num_folds > 1 else evaluation_result.dict(0)
        header = sorted(columns.keys())

        with open_writable_csv_file(self.output_dir, 'evaluation_' + data_type.value) as csv_file:
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
        result.put(TIME_TRAIN, train_time, data_split.get_num_folds(), data_split.get_fold())
        result.put(TIME_PREDICT, predict_time, data_split.get_num_folds(), data_split.get_fold())
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

    def __init__(self, *args: EvaluationOutput):
        super(ClassificationEvaluation, self).__init__(*args)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()

        if is_multilabel(ground_truth):
            hamming_loss = metrics.hamming_loss(ground_truth, predictions)
            result.put(HAMMING_LOSS, hamming_loss, num_folds, fold)
            result.put(HAMMING_ACCURACY, 1 - hamming_loss, num_folds, fold)
            subset_accuracy = metrics.accuracy_score(ground_truth, predictions)
            result.put(SUBSET_ACCURACY, subset_accuracy, num_folds, fold)
            result.put(SUBSET_ZERO_ONE_LOSS, 1 - subset_accuracy, num_folds, fold)
            result.put(MICRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='micro',
                                                                zero_division=1), num_folds, fold)
            result.put(MICRO_RECALL, metrics.recall_score(ground_truth, predictions, average='micro', zero_division=1),
                       num_folds, fold)
            result.put(MICRO_F1, metrics.f1_score(ground_truth, predictions, average='micro', zero_division=1),
                       num_folds, fold)
            result.put(MICRO_JACCARD, metrics.jaccard_score(ground_truth, predictions, average='micro',
                                                            zero_division=1), num_folds, fold)
            result.put(MACRO_RECALL, metrics.recall_score(ground_truth, predictions, average='macro', zero_division=1),
                       num_folds, fold)
            result.put(MACRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='macro',
                                                                zero_division=1), num_folds, fold)
            result.put(MACRO_F1, metrics.f1_score(ground_truth, predictions, average='macro', zero_division=1),
                       num_folds, fold)
            result.put(MACRO_JACCARD, metrics.jaccard_score(ground_truth, predictions, average='macro',
                                                            zero_division=1), num_folds, fold)
            result.put(EX_BASED_PRECISION, metrics.precision_score(ground_truth, predictions, average='samples',
                                                                   zero_division=1), num_folds, fold)
            result.put(EX_BASED_RECALL, metrics.recall_score(ground_truth, predictions, average='samples',
                                                             zero_division=1), num_folds, fold)
            result.put(EX_BASED_F1, metrics.f1_score(ground_truth, predictions, average='samples', zero_division=1),
                       num_folds, fold)
            result.put(EX_BASED_JACCARD, metrics.jaccard_score(ground_truth, predictions, average='samples',
                                                               zero_division=1), num_folds, fold)
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=DTYPE_UINT8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8))
            accuracy = metrics.accuracy_score(ground_truth, predictions)
            result.put(ACCURACY, accuracy, num_folds, fold)
            result.put(ZERO_ONE_LOSS, 1 - accuracy, num_folds, fold)
            result.put(PRECISION, metrics.precision_score(ground_truth, predictions, zero_division=1), num_folds, fold)
            result.put(RECALL, metrics.recall_score(ground_truth, predictions, zero_division=1), num_folds, fold)
            result.put(F1, metrics.f1_score(ground_truth, predictions, zero_division=1), num_folds, fold)
            result.put(JACCARD, metrics.jaccard_score(ground_truth, predictions, zero_division=1), num_folds, fold)


class RankingEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a multi-label ranker according to commonly used ranking measures.
    """

    def __init__(self, *args: EvaluationOutput):
        super(RankingEvaluation, self).__init__(*args)

    def _populate_result(self, data_split: DataSplit, result: EvaluationResult, predictions, ground_truth):
        if is_multilabel(ground_truth):
            num_folds = data_split.get_num_folds()
            fold = data_split.get_fold()
            ground_truth = enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8)
            result.put(RANK_LOSS, metrics.label_ranking_loss(ground_truth, predictions), num_folds, fold)
            result.put(COVERAGE_ERROR, metrics.coverage_error(ground_truth, predictions), num_folds, fold)
            result.put(LABEL_RANKING_AVERAGE_PRECISION,
                       metrics.label_ranking_average_precision_score(ground_truth, predictions), num_folds, fold)
            result.put(DISCOUNTED_CUMULATIVE_GAIN, metrics.dcg_score(ground_truth, predictions), num_folds, fold)
            result.put(NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, metrics.ndcg_score(ground_truth, predictions), num_folds,
                       fold)
