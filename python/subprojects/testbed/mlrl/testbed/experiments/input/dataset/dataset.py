"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets that are part of input data.
"""
from typing import Any, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties
from mlrl.testbed.experiments.input.data import DatasetInputData
from mlrl.testbed.experiments.state import ExperimentState


class InputDataset(DatasetInputData):
    """
    Represents a dataset that is part of input data.
    """

    NAME = 'Ground truth'

    def __init__(self, name: str):
        """
        :param name: The name of the dataset
        """
        super().__init__(properties=Properties(name=self.NAME, file_name=name),
                         context=Context(include_prediction_scope=False))

    @override
    def update_state(self, state: ExperimentState, input_data: Any):
        """
        See :func:`mlrl.testbed.experiments.input.data.InputData.update_state`
        """
        super().update_state(state, input_data)
        state.dataset = input_data
