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

    def __init__(self, source: Source, input_data: Data):
        """
        :param source:      The source, the input data should be read from
        :param input_data:  The input data that should be read
        """
        self.source = source
        self.input_data = input_data

    def exchange(self, state: ExperimentState) -> Optional[Data]:
        """
        See :func:`mlrl.testbed.experiments.data.DataExchange.exchange`
        """
        return self.source.read_from_source(state, self.input_data)
