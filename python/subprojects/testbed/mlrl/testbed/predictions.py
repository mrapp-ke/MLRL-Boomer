"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from typing import Any, Optional

import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.data import ArffMetaData, save_arff_file
from mlrl.testbed.data_sinks import FileSink, LogSink as BaseLogSink
from mlrl.testbed.dataset import Attribute, AttributeType
from mlrl.testbed.format import OPTION_DECIMALS, format_array
from mlrl.testbed.io import SUFFIX_ARFF
from mlrl.testbed.output.converters import TextConverter
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.problem_type import ProblemType
from mlrl.testbed.training_result import TrainingResult


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
    """

    class Predictions(TextConverter):
        """
        Stores predictions and the corresponding ground truth.
        """

        def __init__(self, predictions, ground_truth):
            """
            :param predictions:     The predictions
            :param ground_truth:    The ground truth
            """
            self.predictions = predictions
            self.ground_truth = ground_truth

        def to_text(self, options: Options, **_) -> Optional[str]:
            """
            See :func:`mlrl.testbed.output.converters.TextConverter.to_text`
            """
            decimals = options.get_int(OPTION_DECIMALS, 2)
            text = 'Ground truth:\n\n'
            text += format_array(self.ground_truth, decimals=decimals)
            text += '\n\nPredictions:\n\n'
            text += format_array(self.predictions, decimals=decimals)
            return text

    class LogSink(BaseLogSink):
        """
        Allows to write predictions and the corresponding ground truth to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(BaseLogSink.TitleFormatter('Predictions'), options=options)

    class ArffFileSink(FileSink):
        """
        Allows to write predictions and the corresponding ground truth to an ARFF file.
        """

        def __init__(self, directory: str, options: Options = Options()):
            """
            :param directory: The path to the directory, where the ARFF file should be located
            """
            super().__init__(FileSink.PathFormatter(directory, 'predictions', SUFFIX_ARFF), options)

        # pylint: disable=unused-argument
        def _write_output(self, file_path: str, scope: OutputScope, training_result: Optional[TrainingResult],
                          prediction_result: Optional[PredictionResult], output_data, **_):
            decimals = self.options.get_int(OPTION_DECIMALS, 0)
            ground_truth = output_data.ground_truth
            predictions = output_data.predictions
            nominal_values = None

            if issubclass(predictions.dtype.type, np.integer):
                if scope.problem_type == ProblemType.CLASSIFICATION:
                    attribute_type = AttributeType.NOMINAL
                    nominal_values = [str(value) for value in np.unique(predictions)]
                else:
                    attribute_type = AttributeType.ORDINAL
            else:
                attribute_type = AttributeType.NUMERICAL

                if decimals > 0:
                    predictions = np.around(predictions, decimals=decimals)

            dataset = scope.dataset
            features = []
            outputs = []

            for output in dataset.outputs:
                features.append(Attribute('Ground Truth ' + output.name, attribute_type, nominal_values))
                outputs.append(Attribute('Prediction ' + output.name, attribute_type, nominal_values))

            save_arff_file(file_path, ground_truth, predictions, ArffMetaData(features, outputs))

    def _generate_output_data(self, scope: OutputScope, _: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        if prediction_result:
            return PredictionWriter.Predictions(predictions=prediction_result.predictions, ground_truth=scope.dataset.y)
        return None
