"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
import sys

from dataclasses import replace
from typing import Optional

import numpy as np

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import DatasetOutputData
from mlrl.testbed.util.format import OPTION_DECIMALS


class Predictions(DatasetOutputData):
    """
    Represents predictions and the corresponding ground truth that are part of output data.
    """

    @staticmethod
    def __format_array(array: np.ndarray, decimals: int = 2) -> str:
        """
        Creates and returns a textual representation of an array.

        :param array:       The array
        :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
        :return:            The textual representation that has been created
        """
        if array.dtype.kind == 'f':
            precision = decimals if decimals > 0 else None
            return np.array2string(array, threshold=sys.maxsize, precision=precision, suppress_small=True)
        # pylint: disable=unnecessary-lambda
        return np.array2string(array, threshold=sys.maxsize, formatter={'all': lambda x: str(x)})

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
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        decimals = options.get_int(OPTION_DECIMALS, 2)
        text = 'Ground truth:\n\n'
        text += self.__format_array(self.original_dataset.y, decimals=decimals)
        text += '\n\nPredictions:\n\n'
        text += self.__format_array(self.prediction_dataset.y, decimals=decimals)
        return text

    # pylint: disable=unused-argument
    def to_dataset(self, options: Options, **_) -> Optional[Dataset]:
        """
        See :func:`mlrl.testbed.experiments.output.data.DatasetOutputData.to_dataset`
        """
        return replace(self.original_dataset, x=self.original_dataset.y, y=self.prediction_dataset.y)
