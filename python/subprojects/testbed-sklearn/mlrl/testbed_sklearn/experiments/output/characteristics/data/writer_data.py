"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of datasets to one or several sinks.
"""
from itertools import chain
from typing import List, Optional, override

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import LABEL_CHARACTERISTICS, \
    OUTPUT_CHARACTERISTICS
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_data import FEATURE_CHARACTERISTICS, \
    DataCharacteristics

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.problem_domain import ClassificationProblem
from mlrl.testbed.experiments.state import ExperimentState

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
        def extract_data(self, state: ExperimentState, sinks: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            tabular_output_data = super().extract_data(state, sinks)

            if tabular_output_data:
                table = tabular_output_data.to_table(Options()).to_column_wise_table()
                columns_by_name = {column.header: column for column in table.columns}
                problem_domain = state.problem_domain
                feature_characteristics = FEATURE_CHARACTERISTICS

                if isinstance(problem_domain, ClassificationProblem):
                    output_characteristics = LABEL_CHARACTERISTICS
                else:
                    output_characteristics = OUTPUT_CHARACTERISTICS

                values = [(characteristic, columns_by_name[characteristic.name][0])
                          for characteristic in chain(feature_characteristics, output_characteristics)
                          if characteristic.name in columns_by_name]
                return DataCharacteristics(values)

            return None

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `DataCharacteristicsWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            dataset = state.dataset_as(self, TabularDataset)

            if dataset:
                return DataCharacteristics.from_dataset(problem_domain=state.problem_domain, dataset=dataset)

            return None

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
