"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for parameter tuning.
"""
from abc import ABC, abstractmethod

from mlrl.testbed.io import create_csv_dict_reader
from mlrl.testbed.io import open_readable_csv_file
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
