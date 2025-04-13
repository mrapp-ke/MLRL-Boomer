"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sources, input data may be read from.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, Optional

from mlrl.testbed.experiments.data import FilePath
from mlrl.testbed.experiments.input.data import InputData, TabularInputData
from mlrl.testbed.experiments.state import ExperimentState


class Source(ABC):
    """
    An abstract base class for all sources, input data may be read from.
    """

    @abstractmethod
    def read_from_source(self, state: ExperimentState, input_data: InputData) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to read input data from the source.

        :param state:       The state that should be used to store the input data
        :param input_data:  The input data that should be read
        :return:            The data that has been read
        """


class FileSource(Source, ABC):
    """
    An abstract base class for all sources that read input data from a file.
    """

    def __init__(self, directory: str, suffix: str):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        """
        self.directory = directory
        self.suffix = suffix

    def read_from_source(self, state: ExperimentState, input_data: InputData) -> Optional[Any]:
        context = input_data.get_context(type(self))
        file_path = FilePath(directory=self.directory,
                             file_name=input_data.properties.file_name,
                             suffix=self.suffix,
                             context=context)
        file_path = file_path.resolve(state)
        log.debug('Reading input data from file "%s"...', file_path)
        data = self._read_from_file(file_path, input_data)

        if data:
            input_data.update_state(state, data)

    @abstractmethod
    def _read_from_file(self, file_path: str, input_data: InputData) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to read input data from a specific sink.

        :param file_path:   The path to the file from which the input data should be read
        :param input_data:  The input data that should be read
        """


class TabularFileSource(FileSource, ABC):
    """
    An abstract base class for all classes that allow to read tabular input data from a file.
    """

    def __init__(self, directory: str, suffix: str):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        """
        super().__init__(directory=directory, suffix=suffix)

    def _read_from_file(self, file_path: str, input_data: InputData) -> Optional[Any]:
        return self._read_table_from_file(file_path, input_data)

    @abstractmethod
    def _read_table_from_file(self, file_path: str, input_data: TabularInputData):
        """
        Must be implemented by subclasses in order to read tabular input data from a specific file.

        :param file_path:   The path to the file from which the input data should be read
        :param input_data:  The tabular input data that should be read
        """
