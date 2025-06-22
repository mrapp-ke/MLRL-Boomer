"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.dataset.dataset import TabularOutputDataset

from mlrl.testbed.experiments.output.data import OutputData


class GroundTruthDataset(TabularOutputDataset):
    """
    Represents a ground truth for tabular data that is part of output data.
    """

    def __init__(self, dataset: TabularDataset):
        """
        :param dataset: A tabular dataset
        """
        super().__init__(dataset=dataset,
                         properties=OutputData.Properties(name='Ground truth', file_name='ground_truth'))
