"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets that are part of input data.
"""
import numpy as np

from mlrl.testbed.experiments.data import Context
from mlrl.testbed.experiments.input.data import DatasetInputData


class InputDataset(DatasetInputData):
    """
    Represents a dataset that is part of input data.
    """

    def __init__(self, dataset_name: str):
        """
        :param dataset_name: The name of the dataset
        """
        super().__init__(DatasetInputData.Properties(file_name=dataset_name,
                                                     feature_dtype=np.float32,
                                                     output_dtype=np.uint8),
                         context=Context(include_prediction_scope=False))
