"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sources, input data may be read from.
"""
import logging as log

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, override

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.file_path import FilePath
from mlrl.testbed.experiments.input.data import DatasetInputData, InputData, TabularInputData
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class Source(ABC):
    """
    An abstract base class for all sources, input data may be read from.
    """

    @abstractmethod
    def is_available(self, state: ExperimentState, input_data: InputData) -> bool:
        """
        Must be implemented by subclasses in order to check whether input data is available or not.

        :param state:       The state that should be used to store the input data
        :param input_data:  The input data that should be read
        :return:            True, if the input data is available, False otherwise
        """

    @abstractmethod
    def read_from_source(self, state: ExperimentState, input_data: InputData):
        """
        Must be implemented by subclasses in order to read input data from the source.

        :param state:       The state that should be used to store the input data
        :param input_data:  The input data that should be read
        """


class FileSource(Source, ABC):
    """
    An abstract base class for all sources that read input data from a file.
    """

    def __init__(self, directory: Path, suffix: str):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        """
        self.directory = directory
        self.suffix = suffix

    def _get_file_path(self, state: ExperimentState, input_data: InputData) -> Path:
        """
        May be overridden by subclasses in order to determine the path to the file, the input data should be read from.

        :param state:       The state that should be used to store the input data
        :param input_data:  The input data that should be read
        :return:            The path to the file, the input data should be read from
        """
        return FilePath(directory=self.directory,
                        file_name=input_data.properties.file_name,
                        suffix=self.suffix,
                        context=input_data.context).resolve(state)

    @override
    def is_available(self, state: ExperimentState, input_data: InputData) -> bool:
        return self._get_file_path(state, input_data).is_file()

    @override
    def read_from_source(self, state: ExperimentState, input_data: InputData):
        file_path = self._get_file_path(state, input_data)
        log.debug('Reading input data from file "%s"...', file_path)
        data = self._read_from_file(state, file_path, input_data)

        if data:
            input_data.update_state(state, data)

    @abstractmethod
    def _read_from_file(self, state: ExperimentState, file_path: Path, input_data: InputData) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to read input data from a specific sink.

        :param state:       The state that should be used to store the input data
        :param file_path:   The path to the file from which the input data should be read
        :param input_data:  The input data that should be read
        """


class DatasetFileSource(FileSource, ABC):
    """
    An abstract base class for all classes that allow to read a dataset from a file.
    """

    @override
    def _read_from_file(self, state: ExperimentState, file_path: Path, input_data: InputData) -> Optional[Any]:
        if isinstance(input_data, DatasetInputData):
            return self._read_dataset_from_file(state, file_path, input_data)
        return None

    @abstractmethod
    def _read_dataset_from_file(self, state: ExperimentState, file_path: Path,
                                input_data: DatasetInputData) -> Optional[Dataset]:
        """
        Must be implemented by subclasses in order to read a dataset from a specific file.

        :param state:       The current state of the experiment
        :param file_path:   The path to the file from which the input data should be read
        :param input_data:  The input data that should be read
        :return:            A dataset that has been read from the file
        """


class TabularFileSource(FileSource, ABC):
    """
    An abstract base class for all classes that allow to read tabular input data from a file.
    """

    def __init__(self, directory: Path, suffix: str):
        """
        :param directory:   The path to the directory of the file
        :param suffix:      The suffix of the file
        """
        super().__init__(directory=directory, suffix=suffix)

    @override
    def _read_from_file(self, _: ExperimentState, file_path: Path, input_data: InputData) -> Optional[Any]:
        if isinstance(input_data, TabularInputData):
            return self._read_table_from_file(file_path, input_data)
        return None

    @abstractmethod
    def _read_table_from_file(self, file_path: Path, input_data: TabularInputData) -> Optional[Table]:
        """
        Must be implemented by subclasses in order to read tabular input data from a specific file.

        :param file_path:   The path to the file from which the input data should be read
        :param input_data:  The tabular input data that should be read
        :return:            A table that has been read from the file
        """
