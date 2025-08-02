"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing output data to sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, List, Optional, override

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


class OutputWriter:
    """
    Allows to write output data to one or several sinks.
    """

    def __extract_data(self, extractor: DataExtractor, state: ExperimentState) -> Optional[OutputData]:
        try:
            return extractor.extract_data(state, self.sinks)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            if self.exit_on_error:
                raise error

            log.error('Failed to extract output data from experimental state via extractor of type %s',
                      type(extractor).__name__,
                      exc_info=error)
            return None

    def __write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        try:
            self._write_to_sink(sink, state, output_data)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            if self.exit_on_error:
                raise error

            log.error('Failed to write output data of type "%s" to sink %s',
                      type(output_data).__name__,
                      type(sink).__name__,
                      exc_info=error)

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        self.extractors = list(extractors)
        self.sinks: List[Sink] = []
        self.exit_on_error = True

    def add_sinks(self, *sinks: Sink) -> 'OutputWriter':
        """
        Adds one or several sinks, output data should be written to.

        :param sinks:   The sinks to be added
        :return:        The `OutputWriter` itself
        """
        self.sinks.extend(sinks)
        return self

    def write(self, state: ExperimentState):
        """
        Writes the input data.

        :param state: The current state of the experiment
        """
        sinks = self.sinks

        if sinks:
            extractors = self.extractors

            if extractors:
                output_data = None

                for extractor in extractors:
                    output_data = self.__extract_data(extractor, state)

                    if output_data:
                        break

                if output_data:
                    for sink in sinks:
                        self.__write_to_sink(sink, state, output_data)
            else:
                log.warning('No extractors have been added to output writer of type %s', type(self).__name__)

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        """
        May be overridden by subclasses in order to write output data to a specific sink.

        :param sink:        The sink, the output data should be written to
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the sink
        """
        sink.write_to_sink(state, output_data)

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

    @override
    def __hash__(self) -> int:
        return hash(type(self))
