"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.dataset.dataset import OutputDataset


class PredictionDataset(OutputDataset):
    """
    Represents predictions that are part of output data.
    """

    def __init__(self, dataset: Dataset):
        """
        :param dataset: A dataset
        """
        super().__init__(dataset=dataset, properties=OutputData.Properties(name='Predictions', file_name='predictions'))
