"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing output data to sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import List, Optional

from mlrl.testbed.experiments.connectors import Connector
from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.state import ExperimentState


class DataExtractor(ABC):
    """
    An abstract base class for all classes that extract output data from the state of an experiment.
    """

    @abstractmethod
    def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> Optional[OutputData]:
        """
        Must be implemented by subclasses in order to extract output data from the state of an experiment.

        :param state:   The state from which the output data should be extracted
        :param sinks:   The sinks to which the extracted data should be written
        :return:        The output data that has been extracted or None, if no output data has been extracted
        """


class OutputWriter(Connector, ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks.
    """

    class Session(Connector.Session):
        """
        A session that allows to pass output data to the environment this software runs in.
        """

        def __init__(self, output_writer: 'OutputWriter', state: ExperimentState):
            """
            :param output_writer:   The output writer that has opened this session
            :param state:           The state from which the output data has been generated
            """
            self.output_writer = output_writer
            self.state = state

        def exchange(self) -> Optional[Data]:
            """
            See :func:`mlrl.testbed.experiments.connectors.Connector.Session.exchange`
            """
            output_writer = self.output_writer
            sinks = output_writer.sinks

            if sinks:
                extractors = output_writer.extractors

                if extractors:
                    state = self.state
                    output_data = None

                    for extractor in extractors:
                        output_data = extractor.extract_data(state, sinks)

                        if output_data:
                            break

                    if output_data:
                        for sink in sinks:
                            # pylint: disable=protected-access
                            output_writer._write_to_sink(sink, state, output_data)
                else:
                    log.warning('No extractors have been added to output writer of type %s', type(self).__name__)
            else:
                log.warning('No sinks have been added to output writer of type %s', type(self).__name__)

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        self.extractors = list(extractors)
        self.sinks = []

    def add_sinks(self, *sinks: Sink) -> 'OutputWriter':
        """
        Adds one or several sinks, output data should be written to.

        :param sinks:   The sinks to be added
        :return:        The `OutputWriter` itself
        """
        self.sinks.extend(sinks)
        return self

    def open_session(self, state: ExperimentState) -> Connector.Session:
        """
        See :func:`mlrl.testbed.experiments.connectors.Connector.open_session`
        """
        return OutputWriter.Session(self, state)

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        """
        May be overridden by subclasses in order to write output data to a specific sink.

        :param sink:        The sink, the output data should be written to
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the sink
        """
        sink.write_to_sink(state, output_data)
