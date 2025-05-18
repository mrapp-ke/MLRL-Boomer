"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of datasets to one or several sinks.
"""

from typing import List, Optional

from mlrl.testbed.experiments.dataset_tabular import TabularDataset
from mlrl.testbed.experiments.output.characteristics.data.tabular.characteristics_data import DataCharacteristics
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
            dataset = state.dataset_as(self, TabularDataset)
            return DataCharacteristics(problem_domain=state.problem_domain, dataset=dataset) if dataset else None

    def __init__(self, *extractors: DataExtractor, exit_on_error: bool = True):
        """
        :param extractors:      Extractors that should be used for extracting the output data to be written to the sinks
        :param exit_on_error:   True, if the program should exit when an error occurs while writing the output data,
                                False otherwise
        """
        super().__init__(*extractors, DataCharacteristicsWriter.DefaultExtractor(), exit_on_error=exit_on_error)
