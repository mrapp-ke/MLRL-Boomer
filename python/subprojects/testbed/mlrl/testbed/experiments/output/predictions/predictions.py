"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from dataclasses import replace
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import DatasetOutputData
from mlrl.testbed.util.format import OPTION_DECIMALS, format_array


class Predictions(DatasetOutputData):
    """
    Represents predictions and the corresponding ground truth that are part of output data.
    """

    def __init__(self, original_dataset: Dataset, prediction_dataset: Dataset):
        """
        :param original_dataset:    The original dataset containing the ground truth
        :param prediction_dataset:  A copy of the original dataset, where the ground truth has been replaced with
                                    predictions obtained from a model
        """
        super().__init__(name='Predictions', file_name='predictions')
        self.original_dataset = original_dataset.enforce_dense_outputs()
        self.prediction_dataset = prediction_dataset.enforce_dense_outputs()

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.OutputData.to_text`
        """
        decimals = options.get_int(OPTION_DECIMALS, 2)
        text = 'Ground truth:\n\n'
        text += format_array(self.original_dataset.y, decimals=decimals)
        text += '\n\nPredictions:\n\n'
        text += format_array(self.prediction_dataset.y, decimals=decimals)
        return text

    # pylint: disable=unused-argument
    def to_dataset(self, options: Options, **_) -> Optional[Dataset]:
        """
        See :func:`mlrl.testbed.experiments.output.data.DatasetOutputData.to_dataset`
        """
        return replace(self.original_dataset, x=self.original_dataset.y, y=self.prediction_dataset.y)
