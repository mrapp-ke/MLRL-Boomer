"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sources, input data may be read from.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any

from mlrl.testbed.experiments.input.data import InputData
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
