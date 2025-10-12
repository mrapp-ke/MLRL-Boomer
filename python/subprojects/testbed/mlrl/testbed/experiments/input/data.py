"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input data.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties, TabularProperties
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class InputData(ABC):
    """
    An abstract base class for all classes that represent input data.
    """

    def __init__(self, properties: Properties, context: Context = Context()):
        """
        :param properties:  The properties of the input data
        :param context:     A `Context` to be used by default for finding a suitable input reader this data can be
                            handled by
        """
        self.properties = properties
        self.context = context

    @abstractmethod
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        Updates the state of an experiment based on given input data.

        :param state:       The state to be updated
        :param input_data:  The input data
        """


class DatasetInputData(InputData, ABC):
    """
    An abstract base class for all classes that represent input data that can be converted into a dataset.
    """

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        state.dataset = input_data


class TabularInputData(InputData, ABC):
    """
    An abstract base class for all classes that represent input data that can be converted into a tabular
    representation.
    """

    def __init__(self, properties: TabularProperties, context: Context = Context()):
        """
        :param properties:  The properties of the input data
        :param context:     A `Context` to be used by default for finding a suitable input reader this data can be
                            handled by
        """
        super().__init__(properties=properties, context=context)

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        self._update_state(state, input_data)

    @abstractmethod
    def _update_state(self, state: ExperimentState, table: Table):
        """
        Must be implemented by subclasses in order to update the state of an experiment based on tabular input data.

        :param state:   The state to be updated
        :param table:   A table
        """


class StructuralInputData(InputData, ABC):
    """
    An abstract base class for all classes that represent input data that can be converted into a structural
    representation, e.g., YAML or JSON.
    """

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        self._update_state(state, input_data)

    @abstractmethod
    def _update_state(self, state: ExperimentState, dictionary: Dict[Any, Any]):
        """
        Must be implemented by subclasses in order to update the state of an experiment based on structural input data.

        :param state:       The state to be updated
        :param dictionary:  A dictionary
        """
