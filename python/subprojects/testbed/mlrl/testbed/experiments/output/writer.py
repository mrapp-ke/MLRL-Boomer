"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing output data to sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Optional

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.state import ExperimentState


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks.
    """

    def __init__(self):
        self.sinks = []

    def add_sinks(self, *sinks: Sink) -> 'OutputWriter':
        """
        Adds one or several sinks, output data should be written to.

        :param sinks:   The sinks to be added
        :return:        The `OutputWriter` itself
        """
        self.sinks.extend(sinks)
        return self

    def write_output(self, state: ExperimentState):
        """
        Generates the output data and writes it to all available sinks.

        :param state: The state from which the output data should be generated
        """
        sinks = self.sinks

        if sinks:
            output_data = self._generate_output_data(state)

            if output_data:
                for sink in sinks:
                    self._write_to_sink(sink, state, output_data)
        else:
            log.warning('No sinks have been added to output writer of type %s', type(self).__name__)

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        """
        May be overridden by subclasses in order to write output data to a specific sink.

        :param sink:        The sink, the output data should be written to
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the sink
        """
        sink.write_to_sink(state, output_data)

    @abstractmethod
    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param state:   The state from which the output data should be generated
        :return:        The output data that has been generated or None, if no output data was generated
        """
