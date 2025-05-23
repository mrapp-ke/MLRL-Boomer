"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from mlrl.testbed.experiments.dataset_tabular import TabularDataset
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.dataset.tabular.dataset import TabularOutputDataset


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
