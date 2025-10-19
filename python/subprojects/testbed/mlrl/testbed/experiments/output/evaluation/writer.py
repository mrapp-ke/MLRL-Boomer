"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing evaluation results that have been aggregated across several experiments to one or
several sinks.
"""
from typing import List, Tuple, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties
from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.evaluation.evaluation_result import AggregatedEvaluationResult
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter
from mlrl.testbed.experiments.state import ExperimentState


class AggregatedEvaluationWriter(ResultWriter):
    """
    Allows writing evaluation results that have been aggregated across several experiments to one or several sinks.
    """

    class InputExtractor(DataExtractor):
        """
        Uses an `AggregatedEvaluationResult` that has previously been loaded via an input reader.
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
            input_data = InputData(properties=properties, context=context)
            input_data_key = input_data.get_key(state)
            extra = state.extras.get(input_data_key)

            if isinstance(extra, AggregatedEvaluationResult):
                return [(state, extra)]

            return []

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(
            AggregatedEvaluationWriter.InputExtractor(properties=AggregatedEvaluationResult.PROPERTIES,
                                                      context=AggregatedEvaluationResult.CONTEXT), *extractors)
