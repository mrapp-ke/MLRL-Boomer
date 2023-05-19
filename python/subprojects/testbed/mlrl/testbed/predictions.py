"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np
from mlrl.testbed.data import MetaData, Label, save_arff_file
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.io import SUFFIX_ARFF, get_file_name_per_fold


class PredictionType(Enum):
    """
    Contains all possible types of predictions that may be obtained from a learner.
    """
    BINARY = 'binary'
    SCORES = 'scores'
    PROBABILITIES = 'probabilities'


class PredictionScope(ABC):
    """
    Provides information about whether predictions have been obtained from a global model or incrementally.
    """

    @abstractmethod
    def is_global(self) -> bool:
        """
        Returns whether the predictions have been obtained from a global model or not.

        :return: True, if the predictions have been obtained from a global model, False otherwise
        """
        pass

    @abstractmethod
    def get_model_size(self) -> int:
        """
        Returns the size of the model from which the prediction have been obtained.

        :return: The size of the model or 0, if the predictions have been obtained from a global model
        """
        pass

    @abstractmethod
    def get_file_name(self, name: str) -> str:
        """
        Returns a file name that corresponds to a specific prediction scope.

        :param name:    The name of the file (without suffix)
        :return:        The file name
        """
        pass


class GlobalPrediction(PredictionScope):
    """
    Provides information about predictions that have been obtained from a global model.
    """

    def is_global(self) -> bool:
        return True

    def get_model_size(self) -> int:
        return 0

    def get_file_name(self, name: str) -> str:
        return name


class IncrementalPrediction(PredictionScope):
    """
    Provides information about predictions that have been obtained incrementally.
    """

    def __init__(self, model_size: int):
        """
        :param model_size: The size of the model, the predictions have been obtained from
        """
        self.model_size = model_size

    def is_global(self) -> bool:
        return False

    def get_model_size(self) -> int:
        return self.model_size

    def get_file_name(self, name: str) -> str:
        return name + '_model-size-' + str(self.model_size)


class PredictionOutput(ABC):
    """
    An abstract base class for all outputs, predictions may be written to.
    """

    @abstractmethod
    def write_predictions(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType,
                          prediction_scope: PredictionScope, predictions, ground_truth):
        """
        Writes predictions to the output.

        :param meta_data:           The meta-data of the data set
        :param data_split:          The split of the available data, the predictions and ground truth labels correspond
                                    to
        :param data_type:           Specifies whether the predictions and ground truth labels correspond to the training
                                    or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param predictions:         The predictions
        :param ground_truth:        The ground truth
        """
        pass


class PredictionLogOutput(PredictionOutput):
    """
    Outputs predictions and ground truth labels using the logger.
    """

    def write_predictions(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType,
                          prediction_scope: PredictionScope, predictions, ground_truth):
        text = 'Ground truth:\n\n' + np.array2string(ground_truth, threshold=sys.maxsize) + '\n\nPredictions:\n\n' \
               + np.array2string(predictions, threshold=sys.maxsize, precision=8, suppress_small=True)
        msg = 'Predictions for ' + data_type.value + ' data'

        if not prediction_scope.is_global():
            msg += ' using a model of size ' + str(prediction_scope.get_model_size())

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

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

    def write_predictions(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType,
                          prediction_scope: PredictionScope, predictions, ground_truth):
        file_name = get_file_name_per_fold(prediction_scope.get_file_name(data_type.get_file_name('predictions')),
                                           SUFFIX_ARFF, data_split.get_fold())
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

    def print(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, prediction_scope: PredictionScope,
              predictions, ground_truth):
        """
        :param meta_data:           The meta-data of the data set
        :param data_split:          The split of the available data, the predictions and ground truth labels correspond
                                    to
        :param data_type:           Specifies whether the predictions and ground truth labels correspond to the training
                                    or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param predictions:         A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the predictions
        :param ground_truth:        A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the ground truth labels
        """
        for output in self.outputs:
            output.write_predictions(meta_data, data_split, data_type, prediction_scope, predictions, ground_truth)
