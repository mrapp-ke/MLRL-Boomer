"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing unique label vectors that are contained in a dataset to one or several sinks.
"""

from typing import List, Optional, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vector_histogram import LabelVectorHistogram
from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vectors import LabelVectors

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.state import ExperimentState


class LabelVectorWriter(OutputWriter):
    """
    Allows to write unique label vectors that are contained in a dataset to one or several sinks.
    """

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `LabelVectorWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(self, TabularDataset)
            return LabelVectors(LabelVectorHistogram.from_dataset(dataset)) if dataset else None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, LabelVectorWriter.DefaultExtractor())
