"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List

from mlrl.testbed.characteristics import LabelCharacteristics
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.training import DataPartition, DataType


class PredictionCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of binary predictions may be written to.
    """

    @abstractmethod
    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         characteristics: LabelCharacteristics):
        """
        Writes the characteristics of a data set to the output.

        :param data_split:      The split of the available data, the characteristics correspond to
        :param data_type:       Specifies whether the predictions correspond to the training or test data
        :param characteristics: The characteristics of the predictions
        """
        pass


class PredictionCharacteristicsLogOutput(PredictionCharacteristicsOutput):
    """
    Outputs the characteristics of binary predictions using the logger.
    """

    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         characteristics: LabelCharacteristics):
        msg = 'Prediction characteristics for ' + data_type.value + ' data'

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n'
        msg += 'Labels: ' + str(characteristics.num_labels) + '\n'
        msg += 'Label density: ' + str(characteristics.label_density) + '\n'
        msg += 'Label sparsity: ' + str(1 - characteristics.label_density) + '\n'
        msg += 'Label imbalance ratio: ' + str(characteristics.avg_label_imbalance_ratio) + '\n'
        msg += 'Label cardinality: ' + str(characteristics.avg_label_cardinality) + '\n'
        msg += 'Distinct label vectors: ' + str(characteristics.num_distinct_label_vectors) + '\n'
        log.info(msg)


class PredictionCharacteristicsCsvOutput(PredictionCharacteristicsOutput):
    """
    Writes the characteristics of binary predictions to a CSV file.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         characteristics: LabelCharacteristics):
        columns = {
            'Labels': characteristics.num_labels,
            'Label density': characteristics.label_density,
            'Label sparsity': 1 - characteristics.label_density,
            'Label imbalance ratio': characteristics.avg_label_imbalance_ratio,
            'Label cardinality': characteristics.avg_label_cardinality,
            'Distinct label vectors': characteristics.num_distinct_label_vectors
        }
        header = sorted(columns.keys())
        with open_writable_csv_file(self.output_dir, 'prediction_characteristics_' + data_type.value,
                                    data_split.get_fold()) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)


class PredictionCharacteristicsPrinter:
    """
    A class that allows to print the characteristics of binary predictions.
    """

    def __init__(self, outputs: List[PredictionCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of binary predictions should be written to
        """
        self.outputs = outputs

    def print(self, data_split: DataSplit, data_type: DataType, y):
        """
        :param data_split:  The split of the available data, the characteristics correspond to
        :param data_type:   Specifies whether the predictions correspond to the training or test data
        :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores
                            the predictions
        """
        if len(self.outputs) > 0:
            characteristics = LabelCharacteristics(y)

            for output in self.outputs:
                output.write_prediction_characteristics(data_split, data_type, characteristics)
