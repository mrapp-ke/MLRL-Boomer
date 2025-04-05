"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from typing import Any, Optional

from mlrl.common.config.options import Options
from mlrl.common.data.arrays import enforce_dense

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.util.format import OPTION_DECIMALS, format_array


class Predictions(OutputData):
    """
    Represents predictions and the corresponding ground truth that are part of output data.
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
