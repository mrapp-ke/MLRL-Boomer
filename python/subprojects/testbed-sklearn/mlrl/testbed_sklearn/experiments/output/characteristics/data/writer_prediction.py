"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of binary predictions to one or several sinks.
"""
from typing import List, Optional, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_prediction import \
    PredictionCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.matrix_label import LabelMatrix

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks import Sink
from mlrl.testbed.experiments.output.writer import DataExtractor, OutputWriter
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import ExperimentState


class PredictionCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
    """

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
                return PredictionCharacteristics(problem_domain=state.problem_domain,
                                                 prediction_matrix=LabelMatrix(prediction_result.predictions))

            return None

    def __init__(self, *extractors: DataExtractor):
        """
        :param extractors: Extractors that should be used for extracting the output data to be written to the sinks
        """
        super().__init__(*extractors, PredictionCharacteristicsWriter.DefaultExtractor())
