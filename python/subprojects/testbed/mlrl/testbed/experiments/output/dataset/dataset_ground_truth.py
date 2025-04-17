"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.dataset.dataset import OutputDataset


class GroundTruthDataset(OutputDataset):
    """
    Represents a ground truth that is part of output data.
    """

    def __init__(self, dataset: Dataset):
        """
        :param dataset: A dataset
        """
        super().__init__(dataset=dataset,
                         properties=OutputData.Properties(name='Ground truth', file_name='ground_truth'))
