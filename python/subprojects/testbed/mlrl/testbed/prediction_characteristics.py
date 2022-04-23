"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List

from mlrl.testbed.characteristics import LabelCharacteristics
from mlrl.testbed.io import clear_directory, open_writable_csv_file, create_csv_dict_writer


class PredictionCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of binary predictions may be written to.
    """

    @abstractmethod
    def write_prediction_characteristics(self, experiment_name: str, characteristics: LabelCharacteristics,
                                         total_folds: int, fold: int = None):
        """
        Writes the characteristics of a data set to the output.

        :param experiment_name: The name of the experiment
        :param characteristics: The characteristics of the predictions
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the characteristics should be written or None, if no cross validation
                                is used
        """
        pass


class PredictionCharacteristicsLogOutput(PredictionCharacteristicsOutput):
    """
    Outputs the characteristics of binary predictions using the logger.
    """

    def write_prediction_characteristics(self, experiment_name: str, characteristics: LabelCharacteristics,
                                         total_folds: int, fold: int = None):
        msg = 'Prediction characteristics for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n'
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

    def __init__(self, output_dir: str, clear_dir: bool = True):
        """
        :param output_dir:  The path of the directory, the CSV files should be written to
        :param clear_dir:   True, if the directory, the CSV files should be written to, should be cleared
        """
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_prediction_characteristics(self, experiment_name: str, characteristics: LabelCharacteristics,
                                         total_folds: int, fold: int = None):
        if fold is not None:
            self.__clear_dir_if_necessary()
            columns = {
                'Labels': characteristics.num_labels,
                'Label density': characteristics.label_density,
                'Label sparsity': 1 - characteristics.label_density,
                'Label imbalance ratio': characteristics.avg_label_imbalance_ratio,
                'Label cardinality': characteristics.avg_label_cardinality,
                'Distinct label vectors': characteristics.num_distinct_label_vectors
            }
            header = sorted(columns.keys())
            header.insert(0, 'Approach')
            columns['Approach'] = experiment_name
            with open_writable_csv_file(self.output_dir, 'prediction_characteristics', fold) as csv_file:
                csv_writer = create_csv_dict_writer(csv_file, header)
                csv_writer.writerow(columns)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class PredictionCharacteristicsPrinter:
    """
    A class that allows to print the characteristics of binary predictions.
    """

    def __init__(self, outputs: List[PredictionCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of binary predictions should be written to
        """
        self.outputs = outputs

    def print(self, experiment_name: str, y, current_fold: int, num_folds: int):
        """
        :param experiment_name: The name of the experiment
        :param y:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the predictions
        :param current_fold:    The current fold
        :param num_folds:       The total number of folds
        """
        if len(self.outputs) > 0:
            characteristics = LabelCharacteristics(y)

            for output in self.outputs:
                output.write_prediction_characteristics(experiment_name, characteristics, num_folds,
                                                        current_fold if num_folds > 1 else None)
