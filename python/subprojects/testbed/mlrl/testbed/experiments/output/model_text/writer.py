"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing textual representations of models to one or several sinks.
"""
import logging as log

from typing import List, Optional

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class ModelAsTextWriter(OutputWriter):
    """
    Allows to write textual representations of models to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `ModelAsTextWriter`, by default.
        """

        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            training_result = state.training_result

            if training_result:
                log.error('Unable to extract model from learner of type %s. No suitable extractor available.',
                          type(training_result.learner).__name__)

    def __init__(self, *extractors: DataExtractor, exit_on_error: bool = True):
        """
        :param extractors:      Extractors that should be used for extracting the output data to be written to the sinks
        :param exit_on_error:   True, if the program should exit when an error occurs while writing the output data,
                                False otherwise
        """
        super().__init__(*extractors, ModelAsTextWriter.DefaultExtractor(), exit_on_error=exit_on_error)
