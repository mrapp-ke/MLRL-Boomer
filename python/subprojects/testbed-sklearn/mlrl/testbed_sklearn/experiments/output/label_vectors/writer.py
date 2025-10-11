"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing unique label vectors that are contained in a dataset to one or several sinks.
"""
from typing import List, Tuple, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vector_histogram import LabelVectorHistogram
from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vectors import LabelVectors

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class LabelVectorWriter(ResultWriter):
    """
    Allows to write unique label vectors that are contained in a dataset to one or several sinks.
    """

    class InputExtractor(TabularDataExtractor):
        """
        Uses `TabularInputData` that has previously been loaded via an input reader.
        """

        @override
        def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            result: List[Tuple[ExperimentState, OutputData]] = []

            for extracted_state, tabular_output_data in super().extract_data(state, sinks):
                table = tabular_output_data.to_table(Options()).to_column_wise_table()
                columns_by_name = {column.header: column for column in table.columns}
                column_label_vector = columns_by_name[LabelVectors.COLUMN_LABEL_VECTOR]
                column_frequency = columns_by_name[LabelVectors.COLUMN_FREQUENCY]
                values = [(label_vector, int(frequency))
                          for label_vector, frequency in zip(column_label_vector, column_frequency)]
                result.append((extracted_state, LabelVectors(values)))

            return result

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `LabelVectorWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(self, TabularDataset)

            if dataset:
                return [(state, LabelVectors.from_histogram(LabelVectorHistogram.from_dataset(dataset)))]

            return []

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(LabelVectorWriter.InputExtractor(properties=LabelVectors.PROPERTIES,
                                                          context=LabelVectors.CONTEXT),
                         *extractors,
                         LabelVectorWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=LabelVectors.PROPERTIES, context=LabelVectors.CONTEXT))
