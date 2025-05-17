"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing the ground truth to one or several sinks.
"""
import logging as log

from typing import List, Optional

from mlrl.testbed.experiments.dataset import TabularDataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.dataset.tabular.dataset_ground_truth import GroundTruthDataset
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class GroundTruthWriter(OutputWriter):
    """
    Allows to write the ground truth for tabular data to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `GroundTruthWriter`, by default.
        """

        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset

            if isinstance(dataset, TabularDataset):
                return GroundTruthDataset(dataset)

            log.error('Cannot handle dataset of type %s', type(dataset).__name__)
            return None

    def __init__(self, *extractors: DataExtractor, exit_on_error: bool = True):
        """
        :param extractors:      Extractors that should be used for extracting the output data to be written to the sinks
        :param exit_on_error:   True, if the program should exit when an error occurs while writing the output data,
                                False otherwise
        """
        super().__init__(*extractors, GroundTruthWriter.DefaultExtractor(), exit_on_error=exit_on_error)
