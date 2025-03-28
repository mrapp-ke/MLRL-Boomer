"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.characteristics import OutputCharacteristics
from mlrl.testbed.experiments.output.converters import TableConverter
from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.prediction_scope import PredictionType


class PredictionCharacteristics(TabularOutputData):
    """
    Stores characteristics of predictions.
    """

    def __init__(self, problem_type: ProblemType, array):
        super().__init__('Prediction characteristics', 'prediction_characteristics')
        self.characteristics = OutputCharacteristics(problem_type, array)

    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.converters.TextConverter.to_text`
        """
        return self.characteristics.to_text(options, **kwargs)

    def to_table(self, options: Options, **kwargs) -> Optional[TableConverter.Table]:
        """
        See :func:`mlrl.testbed.experiments.output.converters.TableConverter.to_table`
        """
        return self.characteristics.to_table(options, **kwargs)


class PredictionCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
    """

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        # Prediction characteristics can only be determined in the case of binary predictions...
        prediction_result = state.prediction_result

        if prediction_result and prediction_result.prediction_type == PredictionType.BINARY:
            return PredictionCharacteristics(state.problem_type, prediction_result.predictions)

        return None
