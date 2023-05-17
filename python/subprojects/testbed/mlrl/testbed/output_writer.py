"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
import logging as log

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.io import open_writable_txt_file

OutputData = TypeVar('OutputData')


class OutputWriter(Generic[OutputData], ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks, e.g., the console or
    output files.
    """

    class Sink(Generic[OutputData], ABC):
        """
        An abstract base class for all sinks, output data may be written to.
        """

        @abstractmethod
        def write_output(self, data_split: DataSplit, output_data: OutputData):
            """
            Must be implemented by subclasses in order to write output data to the sink.

            :param data_split:  Information about the split of the available data, the output data corresponds to
            :param output_data: The output data that should be written to the sink
            """
            pass

    class LogSink(Generic[OutputData], Sink[OutputData]):
        """
        Allows to write output data to the console.
        """

        def __init__(self, title: str):
            """
            :param title: A title that is printed before the actual output data
            """
            self.title = title

        def write_output(self, data_split: DataSplit, output_data: OutputData):
            message = self.title

            if data_split.is_cross_validation_used():
                message += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

            message += ':\n\n' + str(output_data)
            log.info(message)

    class TxtSink(Generic[OutputData], Sink[OutputData]):
        """
        Allows to write output data into a text file.
        """

        def __init__(self, output_dir: str, file_name: OutputData):
            """
            :param output_dir:  The path of the directory, where the text file should be located
            :param file_name:   The name of the text file (without suffix)
            """
            self.output_dir = output_dir
            self.file_name = file_name

        def write_output(self, data_split: DataSplit, output_data: OutputData):
            with open_writable_txt_file(self.output_dir, self.file_name, data_split.get_fold()) as txt_file:
                txt_file.write(str(output_data))

    def __init__(self, sinks: List[Sink[OutputData]]):
        """
        :param sinks: A list that contains all sinks, output data should be written to
        """
        self.sinks = sinks

    @abstractmethod
    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner) -> OutputData:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param meta_data:   The meta-data of the data set
        :param x:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                            stores the feature values
        :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores
                            the ground truth labels
        :param data_split:  Information about the split of the available data, the output data corresponds to
        :param learner:     The learner that has been trained
        :return:            The output data that has been generated
        """
        pass

    def write_output(self, meta_data: MetaData, x, y, data_split: DataSplit, learner):
        """
        Generates the output data and writes it to all available sinks.

        :param meta_data:   The meta-data of the data set
        :param x:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                            stores the feature values
        :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores
                            the ground truth labels
        :param data_split:  Information about the split of the available data, the output data corresponds to
        :param learner:     The learner that has been trained
        """
        sinks = self.sinks

        if len(sinks) > 0:
            output_data = self._generate_output_data(meta_data, x, y, data_split, learner)

            for sink in sinks:
                sink.write_output(data_split, output_data)
