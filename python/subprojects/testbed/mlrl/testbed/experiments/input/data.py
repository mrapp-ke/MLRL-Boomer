"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input data.
"""
from abc import ABC, abstractmethod
from typing import Any

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.state import ExperimentState


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
