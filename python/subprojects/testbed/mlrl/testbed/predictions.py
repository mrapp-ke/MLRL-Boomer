"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
import sys
from typing import Any, List, Optional

import numpy as np
from mlrl.common.options import Options
from mlrl.testbed.data import MetaData, Label, save_arff_file
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.io import SUFFIX_ARFF, get_file_name_per_fold
from mlrl.testbed.output_writer import OutputWriter, Formattable
from mlrl.testbed.prediction_scope import PredictionType, PredictionScope


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and corresponding ground truth labels to one or several sinks.
    """

    class Predictions(Formattable):
        """
        Stores predictions and corresponding ground truth labels.
        """

        def __init__(self, predictions, ground_truth):
            """
            :param predictions:     The predictions
            :param ground_truth:    The ground truth labels
            """
            self.predictions = predictions
            self.ground_truth = ground_truth

        def format(self, _: Options) -> str:
            text = 'Ground truth:\n\n'
            text += np.array2string(self.ground_truth, threshold=sys.maxsize)
            text += '\n\nPredictions:\n\n'
            text += np.array2string(self.predictions, threshold=sys.maxsize, precision=8, suppress_small=True)
            return text

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write predictions and corresponding ground truth labels to the console.
        """

        def __init__(self):
            super().__init__(title='Predictions')

    class ArffSink(OutputWriter.Sink):
        """
        Allows to write predictions and corresponding ground truth labels to ARFF files.
        """

        def __init__(self, output_dir: str):
            """
            :param output_dir: The path of the directory, where the ARFF file should be located
            """
            self.output_dir = output_dir

        def write_output(self, meta_data: MetaData, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data):
            file_name = get_file_name_per_fold(prediction_scope.get_file_name(data_type.get_file_name('predictions')),
                                               SUFFIX_ARFF, data_split.get_fold())
            attributes = [Label('Ground Truth ' + label.attribute_name) for label in meta_data.labels]
            labels = [Label('Prediction ' + label.attribute_name) for label in meta_data.labels]
            prediction_meta_data = MetaData(attributes, labels, labels_at_start=False)
            save_arff_file(self.output_dir, file_name, output_data.ground_truth, output_data.predictions,
                           prediction_meta_data)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              prediction_type: Optional[PredictionType], predictions: Optional[Any]) -> Optional[Any]:
        return PredictionWriter.Predictions(predictions=predictions, ground_truth=y)
