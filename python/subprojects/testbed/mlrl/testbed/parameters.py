"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for loading and printing parameter settings that are used by a learner. The parameter settings can be
written to one or several outputs, e.g., to the console or to a file. They can also be loaded from CSV files.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List

from mlrl.testbed.io import create_csv_dict_reader
from mlrl.testbed.io import open_readable_csv_file, open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.training import DataPartition


class ParameterInput(ABC):

    @abstractmethod
    def read_parameters(self, data_partition: DataPartition) -> dict:
        """
        Reads a parameter setting from the input.

        :param data_partition:  Information about the partition of data, the parameter setting corresponds to
        :return:                A dictionary that stores the parameters
        """
        pass


class ParameterCsvInput(ParameterInput):
    """
    Reads parameter settings from CSV files.
    """

    def __init__(self, input_dir: str):
        """
        :param input_dir: The path of the directory, the CSV files should be read from
        """
        self.input_dir = input_dir

    def read_parameters(self, data_partition: DataPartition) -> dict:
        with open_readable_csv_file(self.input_dir, 'parameters', data_partition.get_fold()) as csv_file:
            csv_reader = create_csv_dict_reader(csv_file)
            return dict(next(csv_reader))


class ParameterOutput(ABC):
    """
    An abstract base class for all outputs, parameter settings may be written to.
    """

    @abstractmethod
    def write_parameters(self, data_partition: DataPartition, learner):
        """
        :param data_partition:  Information about the partition of data, the parameter setting corresponds to
        :param learner:         The learner
        """
        pass


class ParameterLogOutput(ParameterOutput):
    """
    Outputs parameter settings using the logger.
    """

    def write_parameters(self, data_partition: DataPartition, learner):
        msg = 'Custom parameters'

        if data_partition.is_cross_validation_used():
            msg += ' (Fold ' + str(data_partition.get_fold() + 1) + ')'

        msg += ':\n\n'
        params = learner.get_params()

        for key in sorted(params):
            value = params[key]

            if value is not None:
                msg += key + ': ' + str(value) + '\n'

        log.info(msg)


class ParameterCsvOutput(ParameterOutput):
    """
    Writes parameter settings to a CSV file.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_parameters(self, data_partition: DataPartition, learner):
        params = learner.get_params()

        for key, value in list(params.items()):
            if value is None:
                del params[key]

        header = sorted(params)

        with open_writable_csv_file(self.output_dir, 'parameters', data_partition.get_fold()) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(params)


class ParameterPrinter:
    """
    A class that allows to print the parameter setting that is used by a learner.
    """

    def __init__(self, outputs: List[ParameterOutput]):
        """
        :param outputs: The outputs, the parameter settings should be written to
        """
        self.outputs = outputs

    def print(self, data_partition: DataPartition, learner):
        """
        :param data_partition:  Information about the partition of data, the characteristics correspond to
        :param learner:         The learner
        """
        for output in self.outputs:
            output.write_parameters(data_partition, learner)
