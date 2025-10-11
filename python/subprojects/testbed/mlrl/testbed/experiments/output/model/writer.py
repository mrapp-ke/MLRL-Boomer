"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing models to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, override

from mlrl.testbed.experiments.input.model import ModelReader
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.model.arguments import ModelOutputDirectoryArguments
from mlrl.testbed.experiments.output.model.model import OutputModel
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ModelWriter(OutputWriter):
    """
    Allows to write models to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `ModelWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            training_result = state.training_result

            if training_result:
                return OutputModel(training_result.learner)

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, ModelWriter.DefaultExtractor())

    @override
    def create_input_reader(self, args: Namespace, input_directory: Path) -> Optional[InputReader]:
        model_load_dir = ModelOutputDirectoryArguments.MODEL_SAVE_DIR.get_value(args)

        if model_load_dir:
            return ModelReader(*self.create_sources(input_directory / model_load_dir))

        return None
