"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing predictions that are part of output data.
"""
from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.testbed_sklearn.experiments.output.dataset.dataset import TabularOutputDataset

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties
from mlrl.testbed.experiments.input.dataset import InputDataset


class GroundTruthDataset(TabularOutputDataset):
    """
    Represents a ground truth for tabular data that is part of output data.
    """

    PROPERTIES = Properties(name=InputDataset.NAME, file_name='ground_truth')

    CONTEXT = Context()

    def __init__(self, dataset: TabularDataset):
        """
        :param dataset: A tabular dataset
        """
        super().__init__(dataset=dataset, properties=self.PROPERTIES, context=self.CONTEXT)
