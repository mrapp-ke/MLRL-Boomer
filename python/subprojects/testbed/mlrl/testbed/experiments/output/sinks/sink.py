"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sinks, output data may be written to.
"""
from abc import ABC, abstractmethod
from os import path

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import get_file_name_per_fold


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

    class PathFormatter:
        """
        Allows to determine the path to the file to which output data is written.
        """

        def __init__(self, directory: str, file_name: str, suffix: str,
                     formatter_options: ExperimentState.FormatterOptions):
            """
            :param directory:           The path to the directory of the file
            :param file_name:           The name of the file
            :param suffix:              The suffix of the file
            :param formatter_options:   The Options to be used by the formatter
            """
            self.directory = directory
            self.file_name = file_name
            self.suffix = suffix
            self.formatter_options = formatter_options

        def format(self, state: ExperimentState) -> str:
            """
            Determines and returns the path to the file to which the output data should be written.

            :param state: The state from which the output data is generated
            """
            file_name = self.file_name

            if self.formatter_options.include_dataset_type:
                file_name = state.dataset.type.get_file_name(file_name)

            if self.formatter_options.include_prediction_scope:
                prediction_result = state.prediction_result

                if prediction_result:
                    file_name = prediction_result.prediction_scope.get_file_name(file_name)

            if self.formatter_options.include_fold:
                file_name = get_file_name_per_fold(file_name, self.suffix, state.fold.index)

            return path.join(self.directory, file_name)

    def __init__(self, directory: str, suffix: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        :param options:     Options to be taken into account
        """
        super().__init__(options)
        self.directory = directory
        self.suffix = suffix

    def write_to_sink(self, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        path_formatter_options = output_data.get_formatter_options(type(self))
        path_formatter = FileSink.PathFormatter(directory=self.directory,
                                                file_name=output_data.file_name,
                                                suffix=self.suffix,
                                                formatter_options=path_formatter_options)
        file_path = path_formatter.format(state)
        self._write_to_file(file_path, state, output_data, **kwargs)

    @abstractmethod
    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to a specific file.

        :param file_path:   The path to the file to which the output data should be written
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the file
        """
