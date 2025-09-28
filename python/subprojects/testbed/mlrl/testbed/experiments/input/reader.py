"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading input data.
"""
from dataclasses import replace

from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.state import ExperimentState


class InputReader:
    """
    Allows to read input data from one or several sources. If multiple sources are used, they are accessed in the given
    order until any input data can successfully be read.
    """

    def __init__(self, input_data: InputData, *sources: Source):
        """
        :param sources:     The sources, the input data should be read from
        :param input_data:  The input data that should be read
        """
        self.sources = list(sources)
        self.input_data = input_data

    def is_available(self, state: ExperimentState) -> bool:
        """
        Checks whether the input data is available or not.

        :param state:   The current state of the experiment
        :return:        True, if the input data is available, False otherwise
        """
        return any(source.is_available(state, self.input_data) for source in self.sources)

    def read(self, state: ExperimentState) -> ExperimentState:
        """
        Reads the input data.

        :param state: The state that should be used to store the input data
        :return:        A copy of the given state that stores the input data
        """
        for source in self.sources:
            new_state = replace(state)

            if source.read_from_source(new_state, self.input_data):
                return new_state

        return state
