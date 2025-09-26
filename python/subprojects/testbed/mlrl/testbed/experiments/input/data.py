"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input data.
"""
from pathlib import Path
from typing import Any, Dict, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties, TabularProperties
from mlrl.testbed.experiments.file_path import FilePath
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table


class InputData:
    """
    Represents input data.
    """

    def __init__(self, properties: Properties, context: Context = Context()):
        """
        :param properties:  The properties of the input data
        :param context:     A `Context` to be used by default for finding a suitable input reader this data can be
                            handled by
        """
        self.properties = properties
        self.context = context

    def update_state(self, state: ExperimentState, input_data: Any):
        """
        Updates the state of an experiment based on given input data.

        :param state:       The state to be updated
        :param input_data:  The input data
        """
        state.extras[self.get_key(state)] = input_data

    def get_key(self, state: ExperimentState) -> str:
        """
        Returns the key that is used to add the input data to the extras of an `ExperimentState`.

        :param state:   The state to be updated
        :return:        The key
        """
        return str(
            FilePath(
                directory=Path(),
                file_name=self.properties.file_name,
                suffix=None,
                context=self.context,
            ).resolve(state))


class TextualInputData(InputData):
    """
    Input data that can be converted into a textual representation.
    """

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        super().update_state(state, input_data)
        self._update_state(state, input_data)

    def _update_state(self, state: ExperimentState, text: str):
        """
        May be overridden by subclasses in order to update the state of an experiment based on textual input data.

        :param state:   The state to be updated
        :param text:    A text
        """


class DatasetInputData(InputData):
    """
    Input data that can be converted into a dataset.
    """

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        super().update_state(state, input_data)
        state.dataset = input_data


class TabularInputData(InputData):
    """
    Input data that can be converted into a tabular representation.
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
        super().update_state(state, input_data)
        self._update_state(state, input_data)

    def _update_state(self, state: ExperimentState, table: Table):
        """
        May be overridden by subclasses in order to update the state of an experiment based on tabular input data.

        :param state:   The state to be updated
        :param table:   A table
        """


class StructuralInputData(InputData):
    """
    An abstract base class for all classes that represent input data that can be converted into a structural
    representation, e.g., YAML or JSON.
    """

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        super().update_state(state, input_data)
        self._update_state(state, input_data)

    def _update_state(self, state: ExperimentState, dictionary: Dict[Any, Any]):
        """
        May be overridden by subclasses in order to update the state of an experiment based on structural input data.

        :param state:       The state to be updated
        :param dictionary:  A dictionary
        """
