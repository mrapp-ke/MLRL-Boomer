"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
from typing import Any, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.characteristics import OutputCharacteristics
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.prediction_scope import PredictionType


class PredictionCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write the characteristics of binary predictions to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Prediction characteristics', options=options)

    class CsvFileSink(OutputWriter.CsvFileSink):
        """
        Allows to write the characteristics of binary predictions to CSV files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='prediction_characteristics', options=options)

    # pylint: disable=unused-argument
    def _generate_output_data(self, scope: OutputScope, learner, prediction_result: Optional[PredictionResult],
                              train_time: float) -> Optional[Any]:
        # Prediction characteristics can only be determined in the case of binary predictions...
        if prediction_result and prediction_result.prediction_type == PredictionType.BINARY:
            return OutputCharacteristics(scope.problem_type, prediction_result.predictions)
        return None
