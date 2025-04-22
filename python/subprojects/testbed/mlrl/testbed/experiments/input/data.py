"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class InputData(Data, ABC):
    """
    An abstract base class for all classes that represent input data.
    """

    @dataclass
    class Properties:
        """
        Properties of input data.

        Attributes:
            file_name: A file name to be used for reading from input files
        """
        file_name: str

    def __init__(self, properties: Properties, default_context: Data.Context = Data.Context()):
        """
        :param properties:      The properties of the input data
        :param default_context: A `Data.Context` to be used by default for finding a suitable input reader this data
                                can be handled by
        """
        super().__init__(default_context=default_context)
        self.properties = properties

    @abstractmethod
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        Updates the state of an experiment based on given input data.

        :param state:       The state to be updated
        :param input_data:  The input data
        """


class TabularInputData(InputData, ABC):
    """
    An abstract base class for all classes that represent input data that can be converted into a tabular
    representation.
    """

    @dataclass
    class Properties(InputData.Properties):
        """
        Properties of tabular input data.

        Attributes:
            has_header: True, if the tabular input data has a header, False otherwise
        """
        has_header: bool

    def __init__(self, properties: Properties, default_context: Data.Context = Data.Context()):
        """

        :param properties:      The properties of the input data
        :param default_context: A `Data.Context` to be used by default for finding a suitable input reader this data
                                can be handled by

        """
        super().__init__(properties, default_context)

    def update_state(self, state: ExperimentState, input_data: Any):
        self._update_state(state, input_data)

    @abstractmethod
    def _update_state(self, state: ExperimentState, table: Table):
        """
        Must be implemented by subclasses in order to update the state of an experiment based on tabular input data.

        :param state:   The state to be updated
        :param table:   A table
        """
