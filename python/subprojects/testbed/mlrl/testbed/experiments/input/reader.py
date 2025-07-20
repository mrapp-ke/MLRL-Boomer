"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading input data.
"""
from dataclasses import replace
from typing import Any, override

from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.state import ExperimentState


class InputReader:
    """
    Allows to read input data from a source.
    """

    def __init__(self, source: Source, input_data: Any):
        """
        :param source:      The source, the input data should be read from
        :param input_data:  The input data that should be read
        """
        self.source = source
        self.input_data = input_data

    def is_available(self, state: ExperimentState) -> bool:
        """
        Checks whether the input data is available or not.

        :param state:   The current state of the experiment
        :return:        True, if the input data is available, False otherwise
        """
        return self.source.is_available(state, self.input_data)

    def read(self, state: ExperimentState) -> ExperimentState:
        """
        Reads the input data.

        :param state: The state that should be used to store the input data
        :return:        A copy of the given state that stores the input data
        """
        new_state = replace(state)
        self.source.read_from_source(new_state, self.input_data)
        return new_state

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

    @override
    def __hash__(self) -> int:
        return hash(type(self))
