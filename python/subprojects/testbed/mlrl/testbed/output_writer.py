"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any
import logging as log

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.io import open_writable_txt_file
from mlrl.common.options import Options


class Formattable(ABC):
    """
    An abstract base class for all classes from which a textual representation can be created.
    """

    @abstractmethod
    def format(self, options: Options) -> str:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """
        pass


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks, e.g., the console or
    output files.
    """

    class Sink(ABC):
        """
        An abstract base class for all sinks, output data may be written to.
        """

        @abstractmethod
        def write_output(self, data_split: DataSplit, output_data):
            """
            Must be implemented by subclasses in order to write output data to the sink.

            :param data_split:  Information about the split of the available data, the output data corresponds to
            :param output_data: The output data that should be written to the sink
            """
            pass

    class LogSink(Sink):
        """
        Allows to write output data to the console.
        """

        def __init__(self, title: str, options: Options = Options()):
            """
            :param title:   A title that is printed before the actual output data
            :param options: Options to be taken into account
            """
            self.title = title
            self.options = options

        def write_output(self, data_split: DataSplit, output_data):
            message = self.title

            if data_split.is_cross_validation_used():
                message += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

            message += ':\n\n' + output_data.format(self.options)
            log.info(message)

    class TxtSink(Sink):
        """
        Allows to write output data into a text file.
        """

        def __init__(self, output_dir: str, file_name: str, options: Options = Options()):
            """
            :param output_dir:  The path of the directory, where the text file should be located
            :param file_name:   The name of the text file (without suffix)
            :param options:     Options to be taken into account
            """
            self.output_dir = output_dir
            self.file_name = file_name
            self.options = options

        def write_output(self, data_split: DataSplit, output_data):
            with open_writable_txt_file(self.output_dir, self.file_name, data_split.get_fold()) as txt_file:
                txt_file.write(output_data.format(self.options))

    def __init__(self, sinks: List[Sink]):
        """
        :param sinks: A list that contains all sinks, output data should be written to
        """
        self.sinks = sinks

    @abstractmethod
    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner) -> Optional[Any]:
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
        :return:            The output data that has been generated or None, if no output data was generated
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

            if output_data is not None:
                for sink in sinks:
                    sink.write_output(data_split, output_data)
