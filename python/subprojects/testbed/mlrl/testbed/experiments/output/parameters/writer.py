"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing algorithmic parameters to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, override

from mlrl.testbed.experiments.input.parameters import ParameterReader
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.parameters.arguments import ParameterOutputDirectoryArguments
from mlrl.testbed.experiments.output.parameters.parameters import OutputParameters
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ParameterWriter(OutputWriter):
    """
    Allows writing algorithmic parameters to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `ParameterWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            return OutputParameters(state.parameters)

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, ParameterWriter.DefaultExtractor())

    @override
    def create_input_reader(self, args: Namespace, input_directory: Path) -> Optional[InputReader]:
        parameter_load_dir = ParameterOutputDirectoryArguments.PARAMETER_SAVE_DIR.get_value(args)

        if parameter_load_dir:
            return ParameterReader(*self.create_sources(input_directory / parameter_load_dir))

        return None
