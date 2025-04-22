"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading input data.
"""
from abc import ABC
from typing import Optional

from mlrl.testbed.experiments.data import Data, DataExchange
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.state import ExperimentState


class InputReader(DataExchange, ABC):
    """
    An abstract base class for all classes that allow to read input data from a source.
    """

    class Session(DataExchange.Session):
        """
        A session that allows to read input data from a file.
        """

        def __init__(self, input_reader: 'InputReader', state: ExperimentState):
            """
            :param input_reader:    The input reader that has opened this session
            :param state:           The state that should be used to store the input data
            """
            self.input_reader = input_reader
            self.state = state

        def exchange(self) -> Optional[Data]:
            """
            See :func:`mlrl.testbed.experiments.data.DataExchange.Session.exchange`
            """
            state = self.state
            input_reader = self.input_reader
            input_data = input_reader.input_data
            return input_reader.source.read_from_source(state, input_data)

    def __init__(self, source: Source, input_data: Data):
        """
        :param source:      The source, the input data should be read from
        :param input_data:  The input data that should be read
        """
        self.source = source
        self.input_data = input_data

    def open_session(self, state: ExperimentState) -> Session:
        """
        See :func:`mlrl.testbed.experiments.data.DataExchange.open_session`
        """
        return InputReader.Session(self, state)
