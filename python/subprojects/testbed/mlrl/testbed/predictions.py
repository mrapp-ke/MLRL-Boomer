"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing the predictions of a model. The predictions can be written to one or several outputs,
e.g., to the console or to a file.
"""
from typing import Any, Optional

from mlrl.common.config.options import Options
from mlrl.common.data.arrays import enforce_dense

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import OutputData, TabularOutputData
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, format_array, format_number


class PredictionWriter(OutputWriter):
    """
    Allows to write predictions and the corresponding ground truth to one or several sinks.
    """

    class Predictions(TabularOutputData):
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

        def to_table(self, options: Options, **_) -> Optional[TabularOutputData.Table]:
            """
            See :func:`mlrl.testbed.experiments.output.data.TabularOutputData.to_table`
            """
            decimals = options.get_int(OPTION_DECIMALS, 2)
            dataset = self.dataset
            ground_truth = dataset.y
            predictions = self.predictions
            rows = []

            for example_index in range(dataset.num_examples):
                columns = {}

                for output_index, output in enumerate(dataset.outputs):
                    columns['GroundTruth ' + output.name] = format_number(ground_truth[example_index, output_index],
                                                                          decimals=decimals)
                    columns['Prediction ' + output.name] = format_number(predictions[example_index, output_index],
                                                                         decimals=decimals)

                rows.append(columns)

            return rows

    def _generate_output_data(self, state: ExperimentState) -> Optional[OutputData]:
        prediction_result = state.prediction_result

        if prediction_result:
            return PredictionWriter.Predictions(dataset=state.dataset, predictions=prediction_result.predictions)

        return None
