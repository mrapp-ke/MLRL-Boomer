"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of binary predictions to one or several sinks.
"""
from typing import List, Optional, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import LABEL_CHARACTERISTICS, \
    OUTPUT_CHARACTERISTICS
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_prediction import \
    PredictionCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_label import LabelMatrix

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, ResultWriter, TabularDataExtractor
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class PredictionCharacteristicsWriter(ResultWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
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

                if isinstance(problem_domain, ClassificationProblem):
                    characteristics = LABEL_CHARACTERISTICS
                else:
                    characteristics = OUTPUT_CHARACTERISTICS

                values = [(characteristic, columns_by_name[characteristic.name][0])
                          for characteristic in characteristics if characteristic.name in columns_by_name]
                return PredictionCharacteristics(values)

            return None

    class DefaultExtractor(DataExtractor):
        """
        The extractor to be used by a `PredictionCharacteristicsWriter`, by default.
        """

        @override
        def extract_data(self, state: ExperimentState, _: List[Sink]) -> Optional[OutputData]:
            """
            See :func:`mlrl.testbed.experiments.output.writer.DataExtractor.extract_data`
            """
            prediction_result = state.prediction_result

            # Prediction characteristics can only be determined in the case of binary predictions...
            if prediction_result and prediction_result.prediction_type == PredictionType.BINARY:
                prediction_matrix = LabelMatrix(prediction_result.predictions)
                return PredictionCharacteristics.from_prediction_matrix(problem_domain=state.problem_domain,
                                                                        prediction_matrix=prediction_matrix)

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(PredictionCharacteristicsWriter.InputExtractor(properties=PredictionCharacteristics.PROPERTIES,
                                                                        context=PredictionCharacteristics.CONTEXT),
                         *extractors,
                         PredictionCharacteristicsWriter.DefaultExtractor(),
                         input_data=TabularInputData(properties=PredictionCharacteristics.PROPERTIES,
                                                     context=PredictionCharacteristics.CONTEXT))
