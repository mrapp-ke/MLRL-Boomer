#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, i.e., to the console or to a file.
"""
import csv
import logging as log
import os
import os.path as path
from typing import List, Dict, Set

import numpy as np
import sklearn.metrics as metrics

# The name of the hamming loss metric
HAMMING_LOSS = 'Hamm. Loss'

# The name of the hamming accuracy metric
HAMMING_ACCURACY = 'Hamm. Acc.'

# The name of the subset accuracy metric
SUBSET_ACCURACY = 'Subs. Acc.'

# The name of the micro-averaged precision metric
MICRO_PRECISION = 'Mi. Prec.'

# The name of the macro-averaged precision metric
MACRO_PRECISION = 'Ma. Prec.'

# The name of the micro-averaged recall metric
MICRO_RECALL = 'Mi. Rec.'

# The name of the macro-averaged recall metric
MACRO_RECALL = 'Ma. Rec.'

# The name of the micro-averaged F1 metric
MICRO_F1 = 'Mi. F1'

# The name of the macro-averaged F1 metric
MACRO_F1 = 'Ma. F1'

# The name of the rank loss metric
RANK_LOSS = 'Rank Loss'


class Evaluation:
    """
    Base class for all classes that evaluate the predictions provided by a classifier or ranker.
    """

    def evaluate(self, experiment_name: str, predictions, ground_truth, current_fold: int, total_folds: int):
        """
        Evaluates the predictions provided by a classifier or ranker.

        :param experiment_name: The name of the experiment
        :param predictions:     The predictions provided by the classifier
        :param ground_truth:    The true labels
        :param current_fold:    The number of the current fold, starting at 0, or 0 if no cross validation is used
        :param total_folds:     The total number of folds or 1, if no cross validation is used
        """
        pass


class EvaluationResult:
    """
    Stores the evaluation results according to different measures.

    Attributes
        measures:   A 'Set' that stores the names of the measures
        results:    A 'List' that stores a 'Dict', mapping scores to the names of measures, for each fold
    """

    measures: Set[str] = set()

    results: List[Dict[str, float]] = None

    def put(self, name: str, score: float, fold: int, total_folds: int):
        """
        Adds a new score according to a specific measure to the evaluation result.

        :param name:        The name of the measure
        :param score:       The score according to the measure
        :param fold:        The fold the score corresponds to
        :param total_folds: The total number of folds
        """

        if self.results is None:
            self.results = [None] * total_folds
        elif len(self.results) != total_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(name)
        values = self.results[fold] if self.results[fold] is not None else dict()
        values[name] = score
        self.results[fold] = values

    def get(self, name: str, fold: int) -> float:
        """
        Returns the score according to a specific measure and fold.

        :param name:    The name of the measure
        :param fold:    The fold the score corresponds to
        :return:        The score
        """

        values = self.results[fold] if self.results is not None else None

        if values is None:
            raise AssertionError('No evaluation results available')

        return values[name]

    def dict(self, fold: int) -> Dict:
        values = self.results[fold].copy() if self.results is not None else None

        if values is None:
            raise AssertionError('No evaluation results available')

        return values

    def avg(self, name: str) -> (float, float):
        """
        Returns the score and standard deviation according to a specific measure averaged over all folds.

        :param name:    The name of the measure
        :return:        A tuple consisting of the averaged score and standard deviation
        """
        values = np.array([self.get(name, i) for i in range(len(self.results))])
        return np.average(values), np.std(values)

    def avg_dict(self) -> Dict:
        result: Dict[str, float] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure)
            result[measure] = score
            result['Std.-dev. ' + measure] = std_dev[0]

        return result


class Output:
    """
    An abstract base class for all outputs, evaluation results may be written to.
    """

    def __init__(self, output_predictions: bool):
        """
        :param output_predictions: True, if predictions provided by a classifier or ranker should be written to the
                                   output, False otherwise
        """
        self.output_predictions = output_predictions

    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        """
        Writes an evaluation result to the output.

        :param experiment_name:     The name of the experiment
        :param evaluation_result:   The evaluation result to be written
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass

    def write_predictions(self, experiment_name: str, predictions, ground_truth, total_folds: int, fold: int = None):
        """
        Writes predictions to the output.

        :param experiment_name: The name of the experiment
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the predictions should be written or None, if no cross validation is
                                used
        """
        pass


class LogOutput(Output):
    """
    Outputs evaluation result using the logger.
    """

    def __init__(self, output_predictions: bool = False):
        super().__init__(output_predictions)

    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        text = ''

        for measure in evaluation_result.measures:
            if len(text) > 0:
                text += '\n'

            if fold is None:
                score, std_dev = evaluation_result.avg(measure)
                text += (measure + ': ' + str(score))

                if total_folds > 1:
                    text += (' ±' + str(std_dev))
            else:
                score = evaluation_result.get(measure, fold)
                text += (measure + ': ' + str(score))

        msg = ('Overall evaluation result for experiment \"' + experiment_name + '\"' if fold is None else
               'Evaluation result for experiment \"' + experiment_name + '\" (Fold ' + str(
                   fold + 1) + ')') + ':\n\n%s\n'
        log.info(msg, text)

    def write_predictions(self, experiment_name: str, predictions, ground_truth, total_folds: int, fold: int = None):
        if self.output_predictions:
            text = 'Ground truth:\n\n' + np.array2string(ground_truth) + '\n\nPredictions:\n\n' + np.array2string(
                predictions)
            msg = ('Predictions for experiment \"' + experiment_name + '\"' if fold is None else
                   'Predictions for experiment \"' + experiment_name + '\" (Fold ' + str(fold + 1) + ')') + ':\n\n%s\n'
            log.info(msg, text)


class CsvOutput(Output):
    """
    Writes evaluation results to CSV files.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True, output_predictions: bool = False):
        """
        :param output_dir:  The path of the directory, the CSV files should be written to
        :param clear_dir:   True, if the directory, the CSV files should be written to, should be cleared
        """
        super().__init__(output_predictions)
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        self.__clear_dir_if_necessary()
        file_name = 'evaluation_overall.csv' if fold is None else ('evaluation_fold_' + str(fold + 1) + '.csv')
        output_file = path.join(self.output_dir, file_name)
        write_mode = 'a' if path.isfile(output_file) else 'w'

        with open(output_file, mode=write_mode) as csv_file:
            columns = evaluation_result.avg_dict() if fold is None else evaluation_result.dict(fold)
            header = sorted(columns.keys())
            header.insert(0, 'Approach')
            columns['Approach'] = experiment_name
            writer = csv.DictWriter(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                    fieldnames=header)

            if write_mode == 'w':
                writer.writeheader()

            writer.writerow(columns)

    def write_predictions(self, experiment_name: str, predictions, ground_truth, total_folds: int, fold: int = None):
        if self.output_predictions:
            self.__clear_dir_if_necessary()

            if fold is None:
                file_name = 'predictions_' + experiment_name + '.csv'
            else:
                file_name = 'predictions_' + experiment_name + '_fold_' + str(fold + 1) + '.csv'

            output_file = path.join(self.output_dir, file_name)

            with open(output_file, mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                num_examples = predictions.shape[0]
                num_labels = predictions.shape[1]
                header = ['Ground Truth L' + str(i + 1) for i in range(num_labels)]
                header.extend(['Prediction L' + str(i + 1) for i in range(num_labels)])
                writer.writerow(header)

                for n in range(num_examples):
                    columns = [ground_truth[n, i] for i in range(num_labels)]
                    columns.extend([predictions[n, i] for i in range(num_labels)])
                    writer.writerow(columns)

    def __clear_dir_if_necessary(self):
        """
        Clears the output dir, if necessary.
        """

        if self.clear_dir:
            CsvOutput.__clear_dir(self.output_dir)
            self.clear_dir = False

    @staticmethod
    def __clear_dir(directory: str):
        """
        Deletes all files contained in a directory (excluding subdirectories).

        :param directory: The directory to be cleared
        """

        for file in os.listdir(directory):
            file_path = path.join(directory, file)

            if path.isfile(file_path):
                os.unlink(file_path)


class AbstractEvaluation(Evaluation):
    """
    An abstract base class for all classes that evaluate the predictions provided by a classifier or ranker and allow to
    write the results to one or several outputs.

    Attributes
        results: A dict that contains the 'EvaluationResult's of different experiments mapped to the experiments' names
    """

    results: Dict[str, EvaluationResult] = {}

    def __init__(self, *args: Output):
        """
        :param outputs:             The outputs the evaluation results should be written to
        """
        self.outputs = args

    def evaluate(self, experiment_name: str, predictions, ground_truth, current_fold: int, total_folds: int):
        result = self.results[experiment_name] if experiment_name in self.results else EvaluationResult()
        self.results[experiment_name] = result
        self._populate_result(result, predictions, ground_truth, current_fold, total_folds)
        self.__write_predictions(experiment_name, predictions, ground_truth, current_fold, total_folds)
        self.__write_evaluation_result(experiment_name, result, current_fold, total_folds)

    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int,
                         total_folds: int):
        pass

    def __write_predictions(self, experiment_name: str, predictions, ground_truth, current_fold: int, total_folds: int):
        """
        Writes predictions to the outputs.

        :param experiment_name: The name of the experiment
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        :param current_fold:    The fold for which the predictions should be written or None, if no cross validation is
                                used
        :param total_folds:     The total number of folds
        """

        for output in self.outputs:
            output.write_predictions(experiment_name, predictions, ground_truth, total_folds,
                                     current_fold if total_folds > 0 else None)

    def __write_evaluation_result(self, experiment_name: str, result: EvaluationResult, current_fold: int,
                                  total_folds: int):
        """
        Writes an evaluation result to the outputs.

        :param experiment_name: The name of the experiment
        :param result:          The evaluation result
        :param current_fold:    The fold for which the results should be written or None, if no cross validation is used
                                or if the overall results, averaged over all folds, should be written
        :param total_folds:     The total number of folds
        """

        if total_folds > 1:
            for output in self.outputs:
                output.write_evaluation_results(experiment_name, result, total_folds, current_fold)

        if (current_fold + 1) == total_folds:
            for output in self.outputs:
                output.write_evaluation_results(experiment_name, result, total_folds)


class ClassificationEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a classifier according to commonly used bipartition measures.

    Attributes
        result: The 'EvaluationResult' that stores scores according to different bipartition measures
    """

    def __init__(self, *args: Output):
        super().__init__(*args)

    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int,
                         total_folds: int):
        result.put(HAMMING_LOSS, metrics.hamming_loss(ground_truth, predictions), current_fold, total_folds)
        result.put(HAMMING_ACCURACY, 1 - metrics.hamming_loss(ground_truth, predictions), current_fold, total_folds)
        result.put(SUBSET_ACCURACY, metrics.accuracy_score(ground_truth, predictions), current_fold, total_folds)
        result.put(MICRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='micro'), current_fold,
                   total_folds)
        result.put(MICRO_RECALL, metrics.recall_score(ground_truth, predictions, average='micro'), current_fold,
                   total_folds)
        result.put(MICRO_F1, metrics.f1_score(ground_truth, predictions, average='micro'), current_fold, total_folds)
        result.put(MACRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='macro'), current_fold,
                   total_folds)
        result.put(MACRO_RECALL, metrics.recall_score(ground_truth, predictions, average='macro'), current_fold,
                   total_folds)
        result.put(MACRO_F1, metrics.f1_score(ground_truth, predictions, average='macro'), current_fold, total_folds)


class RankingEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a ranker according to commonly used ranking measures.

    Attributes
        result: The 'EvaluationResult' that stores scores according to different bipartition measures
    """

    def __init__(self, *args: Output):
        super().__init__(*args)

    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int,
                         total_folds: int):
        result.put(RANK_LOSS, metrics.label_ranking_loss(ground_truth, predictions), current_fold, total_folds)
