"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
from mlrl.common.options import Options
from mlrl.testbed.characteristics import LabelCharacteristics
from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.predictions import PredictionType
from typing import Any, List, Optional


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

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write the characteristics of binary predictions to CSV files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='prediction_characteristics', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              prediction_type: Optional[PredictionType], predictions: Optional[Any]) -> Optional[Any]:
        # Prediction characteristics can only be determined in the case of binary predictions...
        return LabelCharacteristics(predictions) if prediction_type == PredictionType.BINARY else None
