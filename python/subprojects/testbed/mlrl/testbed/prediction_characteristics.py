"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
from typing import Any, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.characteristics import OutputCharacteristics
from mlrl.testbed.output.sinks.sink_csv import CsvFileSink as BaseCsvFileSink
from mlrl.testbed.output.sinks.sink_log import LogSink as BaseLogSink
from mlrl.testbed.output.writer import OutputWriter
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.prediction_scope import PredictionType
from mlrl.testbed.training_result import TrainingResult


class PredictionCharacteristicsWriter(OutputWriter):
    """
    Allows to write the characteristics of binary predictions to one or several sinks.
    """

    class LogSink(BaseLogSink):
        """
        Allows to write the characteristics of binary predictions to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(BaseLogSink.TitleFormatter('Prediction characteristics'), options=options)

    class CsvFileSink(BaseCsvFileSink):
        """
        Allows to write the characteristics of binary predictions to a CSV file.
        """

        def __init__(self, directory: str, options: Options = Options()):
            """
            :param directory: The path to the directory, where the CSV file should be located
            """
            super().__init__(BaseCsvFileSink.PathFormatter(directory, 'prediction_characteristics'), options=options)

    def _generate_output_data(self, scope: OutputScope, _: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        # Prediction characteristics can only be determined in the case of binary predictions...
        if prediction_result and prediction_result.prediction_type == PredictionType.BINARY:
            return OutputCharacteristics(scope.problem_type, prediction_result.predictions)
        return None
