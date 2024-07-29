"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from typing import Any, Optional

import numpy as np

from mlrl.common.options import Options

from mlrl.testbed.data import MetaData, Output, save_arff_file
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import OPTION_DECIMALS, format_array
from mlrl.testbed.io import SUFFIX_ARFF, get_file_name_per_fold
from mlrl.testbed.output_writer import Formattable, OutputWriter
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
    """

    class Predictions(Formattable):
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

        def format(self, options: Options, **_) -> str:
            """
            See :func:`mlrl.testbed.output_writer.Formattable.format`
            """
            decimals = options.get_int(OPTION_DECIMALS, 2)
            text = 'Ground truth:\n\n'
            text += format_array(self.ground_truth, decimals=decimals)
            text += '\n\nPredictions:\n\n'
            text += format_array(self.predictions, decimals=decimals)
            return text

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write predictions and the corresponding ground truth to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Predictions', options=options)

    class ArffSink(OutputWriter.Sink):
        """
        Allows to write predictions and the corresponding ground truth to ARFF files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            """
            :param output_dir: The path of the directory, where the ARFF file should be located
            """
            super().__init__(options=options)
            self.output_dir = output_dir

        # pylint: disable=unused-argument
        def write_output(self, problem_type: ProblemType, meta_data: MetaData, data_split: DataSplit,
                         data_type: Optional[DataType], prediction_scope: Optional[PredictionScope], output_data, **_):
            """
            See :func:`mlrl.testbed.output_writer.OutputWriter.Sink.write_output`
            """
            decimals = self.options.get_int(OPTION_DECIMALS, 0)
            ground_truth = output_data.ground_truth
            predictions = output_data.predictions

            if decimals > 0 and not issubclass(predictions.dtype.type, np.integer):
                predictions = np.around(predictions, decimals=decimals)

            file_name = get_file_name_per_fold(prediction_scope.get_file_name(data_type.get_file_name('predictions')),
                                               SUFFIX_ARFF, data_split.get_fold())
            features = [Output('Ground Truth ' + output.name) for output in meta_data.outputs]
            outputs = [Output('Prediction ' + output.name) for output in meta_data.outputs]
            prediction_meta_data = MetaData(features, outputs, outputs_at_start=False)
            save_arff_file(self.output_dir, file_name, ground_truth, predictions, prediction_meta_data)

    # pylint: disable=unused-argument
    def _generate_output_data(self, problem_type: ProblemType, meta_data: MetaData, x, y, data_split: DataSplit,
                              learner, data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        return PredictionWriter.Predictions(predictions=predictions, ground_truth=y)
