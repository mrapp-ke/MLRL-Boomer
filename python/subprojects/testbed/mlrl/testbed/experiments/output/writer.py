"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing output data to sinks.
"""
import logging as log

from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional, Tuple, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties, TabularProperties
from mlrl.testbed.experiments.input.data import DatasetInputData, InputData, TabularInputData, TextualInputData
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.output.arguments import ResultDirectoryArguments
from mlrl.testbed.experiments.output.data import DatasetOutputData, OutputData, TabularOutputData, TextualOutputData
from mlrl.testbed.experiments.output.policies import OutputErrorPolicy
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.state import ExperimentState


class DataExtractor(ABC):
    """
    An abstract base class for all classes that extract output data from the state of an experiment.
    """

    @abstractmethod
    def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
        """
        Must be implemented by subclasses in order to extract output data from the state of an experiment.

        :param state:   The state from which the output data should be extracted
        :param sinks:   The sinks to which the extracted data should be written
        :return:        A list that contains the output data that has been extracted and the corresponding states
        """


class TextualDataExtractor(DataExtractor):
    """
    Uses `TextualInputData` that has previously been loaded via an input reader.
    """

    def __init__(self, properties: Properties, context: Context):
        """
        :param properties:  The properties of the input data
        :param context:     The context of the input data
        """
        self.properties = properties
        self.context = context

    @override
    def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        properties = self.properties
        context = self.context
        input_data = TextualInputData(properties=properties, context=context)
        input_data_key = input_data.get_key(state)
        extra = state.extras.get(input_data_key)

        if extra:
            return [(state, TextualOutputData.from_text(properties=properties, context=context, text=extra))]

        return []


class TabularDataExtractor(DataExtractor):
    """
    Uses `TabularInputData` that has previously been loaded via an input reader.
    """

    def __init__(self, properties: TabularProperties, context: Context):
        """
        :param properties:  The properties of the input data
        :param context:     The context of the input data
        """
        self.properties = properties
        self.context = context

    @override
    def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        properties = self.properties
        context = self.context
        input_data = TabularInputData(properties=properties, context=context)
        input_data_key = input_data.get_key(state)
        extra = state.extras.get(input_data_key)

        if extra:
            return [(state, TabularOutputData.from_table(properties=properties, context=context, table=extra))]

        return []


class DatasetExtractor(DataExtractor, ABC):
    """
    An abstract base class for all extractors that use `DatasetInputData` that has previously been loaded via an input
    reader.
    """

    def __init__(self, properties: Properties, context: Context):
        """
        :param properties:  The properties of the input data
        :param context:     The context of the input data
        """
        self.properties = properties
        self.context = context

    @override
    def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
        """
        See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
        """
        properties = self.properties
        context = self.context
        input_data = DatasetInputData(properties=properties, context=context)
        input_data_key = input_data.get_key(state)
        extra = state.extras.get(input_data_key)

        if extra:
            dataset_output_data = self._create_output_data(extra)

            if dataset_output_data:
                return [(state, dataset_output_data)]

        return []

    @abstractmethod
    def _create_output_data(self, data: Any) -> Optional[DatasetOutputData]:
        """
        Must be implemented by subclasses in order to create output data from given data that has been read via in input
        reader.

        :param data:    The data that has been returned by the input reader
        :return:        The output data that has been created or None, if no output data has been created
        """


class OutputWriter:
    """
    Allows to write output data to one or several sinks.
    """

    def __extract_data_from_extractor(self, extractor: DataExtractor,
                                      state: ExperimentState) -> List[Tuple[ExperimentState, OutputData]]:
        try:
            return extractor.extract_data(state, self.sinks)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            if self.output_error_policy == OutputErrorPolicy.EXIT:
                raise error

            log.error('Failed to extract output data from experimental state via extractor of type %s',
                      type(extractor).__name__,
                      exc_info=error)
            return []

    def __extract_data(self, state: ExperimentState) -> List[Tuple[ExperimentState, OutputData]]:
        extractors = self.extractors

        if extractors:
            for extractor in extractors:
                result = self.__extract_data_from_extractor(extractor, state)

                if result:
                    return result
        else:
            log.warning('No extractors have been added to output writer of type %s', type(self).__name__)

        return []

    def __write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        try:
            self._write_to_sink(sink, state, output_data)
        # pylint: disable=broad-exception-caught
        except Exception as error:
            if self.output_error_policy == OutputErrorPolicy.EXIT:
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
        self.output_error_policy = OutputErrorPolicy.EXIT

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
            for output_state in self._create_states(state):
                for extracted_state, output_data in self.__extract_data(output_state):
                    for sink in sinks:
                        self.__write_to_sink(sink, extracted_state, output_data)

    def _create_states(self, state: ExperimentState) -> List[ExperimentState]:
        """
        May be overridden by subclasses in order create a list of states from which output data should be extracted.

        :param state:   The current state of the experiment
        :return:        A list that contains the states from which output data should be extracted
        """
        return [state]

    def _write_to_sink(self, sink: Sink, state: ExperimentState, output_data: OutputData):
        """
        May be overridden by subclasses in order to write output data to a specific sink.

        :param sink:        The sink, the output data should be written to
        :param state:       The state from which the output data has been generated
        :param output_data: The output data that should be written to the sink
        """
        sink.write_to_sink(state, output_data)

    def create_sources(self, input_directory: Path) -> List[Source]:
        """
        Creates and returns a list that contains all sources that can read the data produced by this output writer.

        :param input_directory: The directory, the data should be read from
        :return:                A list that contains the sources that has been created
        """
        return list(filter(None, map(lambda sink: sink.create_source(input_directory), self.sinks)))

    @abstractmethod
    def create_input_reader(self, args: Namespace, input_directory: Path) -> Optional[InputReader]:
        """
        May be overridden by subclasses in order to create an `InputReader` that can read the data produced by this
        output writer.

        :param args:            The command line arguments specified by the user
        :param input_directory: The directory, the data should be read from
        :return:                The `InputReader` that has been created or None, if no such reader is available
        """
        return None

    @override
    def __str__(self) -> str:
        return type(self).__name__

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

    @override
    def __hash__(self) -> int:
        return hash(type(self))


class ResultWriter(OutputWriter):
    """
    Allows to write experimental results to one or several sinks.
    """

    def __init__(self, *extractors: DataExtractor, input_data: Optional[InputData] = None):
        """
        :param extractors:  Extractors that should be used for extracting the output data to be written to the sinks
        :param input_data:  The `InputData` that corresponds to the output data written by this writer or None, if no
                            such input data is available
        """
        super().__init__(*extractors)
        self.input_data = input_data

    @override
    def create_input_reader(self, args: Namespace, input_directory: Path) -> Optional[InputReader]:
        input_data = self.input_data

        if input_data:
            result_dir = ResultDirectoryArguments.RESULT_DIR.get_value(args)

            if result_dir:
                sources = self.create_sources(input_directory / result_dir)
                return InputReader(input_data, *sources)

        return None
