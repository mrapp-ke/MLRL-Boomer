"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for writing characteristics of binary predictions to one or several sinks.
"""
from typing import Optional

from mlrl.testbed.experiments.output.characteristics.characteristics_prediction import PredictionCharacteristics
from mlrl.testbed.experiments.output.characteristics.matrix_label import LabelMatrix
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.prediction_scope import PredictionType


class PredictionCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
    """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        prediction_result = state.prediction_result

        # Prediction characteristics can only be determined in the case of binary predictions...
        if prediction_result and prediction_result.prediction_type == PredictionType.BINARY:
            return PredictionCharacteristics(state.problem_type, LabelMatrix(prediction_result.predictions))

        return None
