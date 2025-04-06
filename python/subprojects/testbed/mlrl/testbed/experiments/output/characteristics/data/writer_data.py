"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of datasets to one or several sinks.
"""
from typing import List, Optional

from mlrl.testbed.experiments.output.characteristics.data.characteristics_data import DataCharacteristics
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class DataCharacteristicsWriter(OutputWriter):
    """
    Allows writing characteristics of a dataset to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `DataCharacteristicsWriter`, by default.
        """

        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            return DataCharacteristics(problem_type=state.problem_type, dataset=state.dataset)

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, DataCharacteristicsWriter.DefaultExtractor())
