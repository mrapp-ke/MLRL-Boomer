"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets that are part of input data.
"""
from mlrl.common.data.types import Float32, Uint8

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.input.data import DatasetInputData


class InputDataset(DatasetInputData):
    """
    Represents a dataset that is part of input data.
    """

    def __init__(self, dataset_name: str):
        """
        :param dataset_name: The name of the dataset
        """
        super().__init__(DatasetInputData.Properties(file_name=dataset_name, feature_dtype=Float32, output_dtype=Uint8),
                         default_context=Data.Context(include_prediction_scope=False))
