"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of datasets to one or several sinks.
"""
from itertools import chain
from typing import List, Tuple, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import get_output_characteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_data import FEATURE_CHARACTERISTICS, \
    DataCharacteristics

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import parse_number

from mlrl.util.options import Options


class DataCharacteristicsWriter(ResultWriter):
    """
    Allows writing characteristics of a dataset to one or several sinks.
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
                feature_characteristics = FEATURE_CHARACTERISTICS
                output_characteristics = get_output_characteristics(extracted_state.problem_domain)
                values = [(characteristic,
                           parse_number(columns_by_name[characteristic.name][0], percentage=characteristic.percentage))
                          for characteristic in chain(feature_characteristics, output_characteristics)
                          if characteristic.name in columns_by_name]
                result.append((extracted_state, DataCharacteristics(values)))

            return result

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `DataCharacteristicsWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> List[Tuple[ExperimentState, OutputData]]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(TabularDataset)

            if dataset:
                return [(state, DataCharacteristics.from_dataset(problem_domain=state.problem_domain, dataset=dataset))]

            return []

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(DataCharacteristicsWriter.InputExtractor(properties=DataCharacteristics.PROPERTIES,
                                                                  context=DataCharacteristics.CONTEXT),
                         *extractors,
                         DataCharacteristicsWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=DataCharacteristics.PROPERTIES,
                                                     context=DataCharacteristics.CONTEXT))
