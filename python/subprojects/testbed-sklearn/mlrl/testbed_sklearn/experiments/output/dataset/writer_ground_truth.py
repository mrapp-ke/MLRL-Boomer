"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing the ground truth to one or several sinks.
"""
from typing import Any, List, Optional, Tuple, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.dataset.dataset_ground_truth import GroundTruthDataset

from mlrl.testbed.experiments.input.data import DatasetInputData
from mlrl.testbed.experiments.output.data import DatasetOutputData, OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, DatasetExtractor, ResultWriter
from mlrl.testbed.experiments.state import ExperimentState


class GroundTruthWriter(ResultWriter):
    """
    Allows to write the ground truth for tabular data to one or several sinks.
    """

    class GroundTruthExtractor(DatasetExtractor):
        """
        Uses `DatasetInputData` that has previously been loaded via an input reader.
        """

        @override
        def _create_output_data(self, data: Any) -> Optional[DatasetOutputData]:
            return GroundTruthDataset(data)

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `GroundTruthWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(TabularDataset)

            if dataset:
                return [(state, GroundTruthDataset(dataset))]

            return []

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(GroundTruthWriter.GroundTruthExtractor(properties=GroundTruthDataset.PROPERTIES,
                                                                context=GroundTruthDataset.CONTEXT),
                         *extractors,
                         GroundTruthWriter.DefaultExtractor(),
                         input_data=DatasetInputData(properties=GroundTruthDataset.PROPERTIES,
                                                     context=GroundTruthDataset.CONTEXT))
