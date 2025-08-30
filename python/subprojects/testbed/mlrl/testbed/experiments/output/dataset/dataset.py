"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets that are part of output data.
"""

from abc import ABC
from typing import Optional, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.output.data import DatasetOutputData, OutputData

from mlrl.util.options import Options


class OutputDataset(DatasetOutputData, ABC):
    """
    An abstract base class for all classes that represent a dataset that is part of output data.
    """

    def __init__(self, dataset: Dataset, properties: OutputData.Properties, context: Context = Context()):
        """
        :param dataset:     A dataset
        :param properties:  The properties of the output data
        :param context:     A `Context` to be used by default for finding a suitable sink this output data can be
                            written to
        """
        super().__init__(properties=properties, context=context)
        self.dataset = dataset

    @override
    def to_dataset(self, options: Options, **_) -> Optional[Dataset]:
        """
        See :func:`mlrl.testbed.experiments.output.data.DatasetOutputData.to_dataset`
        """
        return self.dataset
