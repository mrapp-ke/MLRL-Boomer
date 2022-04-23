"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from mlrl.testbed.data import MetaData, Label, save_arff_file
from mlrl.testbed.io import clear_directory, SUFFIX_ARFF, get_file_name_per_fold


class PredictionOutput(ABC):
    """
    An abstract base class for all outputs, predictions may be written to.
    """

    @abstractmethod
    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        """
        Writes predictions to the output.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the data set
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the predictions should be written or None, if no cross validation is
                                used
        """
        pass


class PredictionLogOutput(PredictionOutput):
    """
    Outputs predictions and ground truth labels using the logger.
    """

    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        text = 'Ground truth:\n\n' + np.array2string(ground_truth, threshold=sys.maxsize) + '\n\nPredictions:\n\n' \
               + np.array2string(predictions, threshold=sys.maxsize)
        msg = ('Predictions for experiment \"' + experiment_name + '\"' if fold is None else
               'Predictions for experiment \"' + experiment_name + '\" (Fold ' + str(fold + 1) + ')') + ':\n\n%s\n'
        log.info(msg, text)


class PredictionArffOutput(PredictionOutput):
    """
    Writes predictions and ground truth labels to ARFF files.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True):
        """
        :param output_dir:  The path of the directory, the CSV files should be written to
        :param clear_dir:   True, if the directory, the CSV files should be written to, should be cleared
        """
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        self.__clear_dir_if_necessary()
        file_name = get_file_name_per_fold('predictions_' + experiment_name, SUFFIX_ARFF, fold)
        attributes = [Label('Ground Truth ' + label.attribute_name) for label in meta_data.labels]
        labels = [Label('Prediction ' + label.attribute_name) for label in meta_data.labels]
        prediction_meta_data = MetaData(attributes, labels, labels_at_start=False)
        save_arff_file(self.output_dir, file_name, ground_truth, predictions, prediction_meta_data)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class PredictionPrinter:
    """
    A class that allows to print predictions and ground truth labels.
    """

    def __init__(self, outputs: List[PredictionOutput]):
        """
        :param outputs: The outputs, the characteristics of data sets should be written to
        """
        self.outputs = outputs

    def print(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, current_fold: int,
              num_folds: int):
        """
        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the data set
        :param predictions:     A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the predictions
        :param ground_truth:    A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
        :param current_fold:    The current fold
        :param num_folds:       The total number of folds
        """
        for output in self.outputs:
            output.write_predictions(experiment_name, meta_data, predictions, ground_truth, num_folds,
                                     current_fold if num_folds > 1 else None)
