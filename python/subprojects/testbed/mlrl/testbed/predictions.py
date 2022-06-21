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
from mlrl.testbed.io import SUFFIX_ARFF, get_file_name_per_fold
from mlrl.testbed.training import DataPartition, DataType


class PredictionOutput(ABC):
    """
    An abstract base class for all outputs, predictions may be written to.
    """

    @abstractmethod
    def write_predictions(self, meta_data: MetaData, data_partition: DataPartition, data_type: DataType, predictions,
                          ground_truth):
        """
        Writes predictions to the output.

        :param meta_data:       The meta data of the data set
        :param data_partition:  The partition of data, the predictions and ground truth labels correspond to
        :param data_type:       Specifies whether the predictions and ground truth labels correspond to the training or
                                test data
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        """
        pass


class PredictionLogOutput(PredictionOutput):
    """
    Outputs predictions and ground truth labels using the logger.
    """

    def write_predictions(self, meta_data: MetaData, data_partition: DataPartition, data_type: DataType, predictions,
                          ground_truth):
        text = 'Ground truth:\n\n' + np.array2string(ground_truth, threshold=sys.maxsize) + '\n\nPredictions:\n\n' \
               + np.array2string(predictions, threshold=sys.maxsize, precision=8, suppress_small=True)
        msg = 'Predictions for ' + data_type.value + ' data'

        if data_partition.is_cross_validation_used():
            msg += ' (Fold ' + str(data_partition.get_fold() + 1) + ')'

        msg += ':\n\n%s\n'
        log.info(msg, text)


class PredictionArffOutput(PredictionOutput):
    """
    Writes predictions and ground truth labels to ARFF files.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_predictions(self, meta_data: MetaData, data_partition: DataPartition, data_type: DataType, predictions,
                          ground_truth):
        file_name = get_file_name_per_fold('predictions_' + data_type.value, SUFFIX_ARFF, data_partition.get_fold())
        attributes = [Label('Ground Truth ' + label.attribute_name) for label in meta_data.labels]
        labels = [Label('Prediction ' + label.attribute_name) for label in meta_data.labels]
        prediction_meta_data = MetaData(attributes, labels, labels_at_start=False)
        save_arff_file(self.output_dir, file_name, ground_truth, predictions, prediction_meta_data)


class PredictionPrinter:
    """
    A class that allows to print predictions and ground truth labels.
    """

    def __init__(self, outputs: List[PredictionOutput]):
        """
        :param outputs: The outputs, the characteristics of data sets should be written to
        """
        self.outputs = outputs

    def print(self, meta_data: MetaData, data_partition: DataPartition, data_type: DataType, predictions, ground_truth):
        """
        :param meta_data:       The meta data of the data set
        :param data_partition:  The partition of data, the predictions and ground truth labels correspond to
        :param data_type:       Specifies whether the predictions and ground truth labels correspond to the training or
                                test data
        :param predictions:     A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the predictions
        :param ground_truth:    A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
        """
        for output in self.outputs:
            output.write_predictions(meta_data, data_partition, data_type, predictions, ground_truth)
