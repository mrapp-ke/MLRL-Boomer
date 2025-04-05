"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from typing import Any, Optional

import numpy as np

from mlrl.common.config.options import Options
from mlrl.common.data.arrays import enforce_dense

from mlrl.testbed.data import ArffMetaData, save_arff_file
from mlrl.testbed.dataset import Attribute, AttributeType, Dataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, format_array
from mlrl.testbed.util.io import SUFFIX_ARFF


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
    """

    class Predictions(OutputData):
        """
        Stores predictions and the corresponding ground truth.
        """

        def __init__(self, dataset: Dataset, predictions: Any):
            """
            :param dataset:     The dataset for which the predictions have been obtained
            :param predictions: A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the predictions
            """
            super().__init__('Predictions', 'predictions')
            self.dataset = dataset.enforce_dense_outputs()
            self.predictions = enforce_dense(predictions, order='C', dtype=predictions.dtype)

        def to_text(self, options: Options, **_) -> Optional[str]:
            """
            See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
            """
            decimals = options.get_int(OPTION_DECIMALS, 2)
            text = 'Ground truth:\n\n'
            text += format_array(self.dataset.y, decimals=decimals)
            text += '\n\nPredictions:\n\n'
            text += format_array(self.predictions, decimals=decimals)
            return text

    class ArffFileSink(FileSink):
        """
        Allows to write predictions and the corresponding ground truth to an ARFF file.
        """

        def __init__(self, directory: str, options: Options = Options()):
            """
            :param directory:   The path to the directory, where the ARFF file should be located
            :param options:     Options to be taken into account
            """
            super().__init__(directory=directory, suffix=SUFFIX_ARFF)
            self.options = options

        def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **_):
            decimals = self.options.get_int(OPTION_DECIMALS, 0)
            predictions = output_data.predictions
            nominal_values = None

            if issubclass(predictions.dtype.type, np.integer):
                if state.problem_type == ProblemType.CLASSIFICATION:
                    attribute_type = AttributeType.NOMINAL
                    nominal_values = [str(value) for value in np.unique(predictions)]
                else:
                    attribute_type = AttributeType.ORDINAL
            else:
                attribute_type = AttributeType.NUMERICAL

                if decimals > 0:
                    predictions = np.around(predictions, decimals=decimals)

            dataset = output_data.dataset
            features = []
            outputs = []

            for output in dataset.outputs:
                features.append(Attribute('Ground Truth ' + output.name, attribute_type, nominal_values))
                outputs.append(Attribute('Prediction ' + output.name, attribute_type, nominal_values))

            save_arff_file(file_path, dataset.y, predictions, ArffMetaData(features, outputs))

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        prediction_result = state.prediction_result

        if prediction_result:
            return PredictionWriter.Predictions(dataset=state.dataset, predictions=prediction_result.predictions)

        return None
