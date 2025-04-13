"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sinks, output data may be written to.
"""
import logging as log

from abc import ABC, abstractmethod

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.data import FilePath
from mlrl.testbed.experiments.output.data import DatasetOutputData, OutputData, TabularOutputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class Sink(ABC):
    """
    An abstract base class for all sinks, output data may be written to.
    """

    def __init__(self, options: Options = Options()):
        """
        :param options: Options to be taken into account
        """
        self.options = options

    @abstractmethod
    def write_to_sink(self, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to the sink.

        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the sink
        """


class FileSink(Sink, ABC):
    """
    An abstract base class for all sinks that write output data to a file.
    """

    def __init__(self, directory: str, suffix: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        """
        super().__init__(options)
        self.directory = directory
        self.suffix = suffix

    def write_to_sink(self, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        context = output_data.get_context(type(self))
        file_path = FilePath(directory=self.directory,
                             file_name=output_data.properties.file_name,
                             suffix=self.suffix,
                             context=context)
        file_path = file_path.resolve(state)
        log.debug('Writing output data to file "%s"...', file_path)
        self._write_to_file(file_path, state, output_data, **kwargs)

    @abstractmethod
    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to a specific file.

        :param file_path:   The path to the file to which the output data should be written
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the file
        """


class TabularFileSink(FileSink, ABC):
    """
    An abstract base class for all sinks that write tabular output data to a file.
    """

    def __init__(self, directory: str, suffix: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=suffix, options=options)

    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **kwargs):
        if not isinstance(output_data, TabularOutputData):
            raise RuntimeError('Output data of type "' + type(output_data).__name__
                               + '" cannot be converted into a tabular representation')

        tabular_data = output_data.to_table(self.options, **kwargs)

        if tabular_data:
            self._write_table_to_file(file_path, state, tabular_data, **kwargs)

    def _write_table_to_file(self, file_path: str, state: ExperimentState, table: Table, **kwargs):
        """
        Must be implemented by subclasses in order to write tabular output data to a specific file.

        :param file_path:   The path to the file to which the output data should be written
        :param state:       The state from which the output data has been generated
        :param table:       The tabular output data that should be written to the file
        """


class DatasetFileSink(FileSink, ABC):
    """
    An abstract base class for all sinks that write datasets to a file.
    """

    def __init__(self, directory: str, suffix: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=suffix, options=options)

    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **kwargs):
        if not isinstance(output_data, DatasetOutputData):
            raise RuntimeError('Output data of type "' + type(output_data).__name__
                               + '" cannot be converted into a dataset')

        dataset = output_data.to_dataset(self.options, **kwargs)

        if dataset:
            self._write_dataset_to_file(file_path, state, dataset, **kwargs)

    def _write_dataset_to_file(self, file_path: str, state: ExperimentState, dataset: Dataset, **kwargs):
        """
        Must be implemented by subclasses in order to write a dataset to a specific file.

        :param file_path:   The path to the file to which the output data should be written
        :param state:       The state from which the output data has been generated
        :param dataset:     The dataset that should be written to the file
        """
