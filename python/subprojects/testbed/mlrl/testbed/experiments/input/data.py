"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input data.
"""
from abc import ABC, abstractmethod
from typing import Any

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class InputData(Data, ABC):
    """
    An abstract class for all classes that represent input data that can be read.
    """

    @abstractmethod
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        Updates the state of an experiment based on given input data.

        :param state:       The state to be updated
        :param input_data:  The input data
        """


class TabularInputData(InputData, ABC):
    """
    An abstract class for all classes that represent input data that can be converted into a tabular representation.
    """

    def __init__(self, has_header: bool, default_context: Data.Context = Data.Context()):
        """
        :param default_context: A `Data.Context` to be used by default for finding a suitable input reader this data
                                can be handled by
        :param has_header:      True, if the tabular input data has a header, False otherwise

        """
        super().__init__(default_context)
        self.has_header = has_header

    def update_state(self, state: ExperimentState, input_data: Any):
        self._update_state(state, input_data)

    @abstractmethod
    def _update_state(self, state: ExperimentState, table: Table):
        """
        Must be implemented by subclasses in order to update the state of an experiment based on tabular input data.

        :param state:   The state to be updated
        :param table:   A table
        """
