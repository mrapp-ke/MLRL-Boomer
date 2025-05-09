"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets that are part of output data.
"""
import sys

from typing import Optional

import numpy as np

from mlrl.testbed.experiments.data import Data
from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.output.data import DatasetOutputData, OutputData
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.options import Options


class OutputDataset(DatasetOutputData):
    """
    Represents a dataset that is part of output data.
    """

    def __init__(self, dataset: Dataset, properties: OutputData.Properties, context: Data.Context = Data.Context()):
        """
        :param dataset:     A dataset
        :param properties:  The properties of the output data
        :param context:     A `Data.Context` to be used by default for finding a suitable sink this output data can be
                            written to
        """
        super().__init__(properties=properties, context=context)
        self.dataset = dataset.enforce_dense_outputs()

    def to_text(self, options: Options, **_) -> Optional[str]:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        y = self.dataset.enforce_dense_outputs().y

        if y.dtype.kind == 'f':
            decimals = options.get_int(OPTION_DECIMALS, 2)
            precision = decimals if decimals > 0 else None
            return np.array2string(y, threshold=sys.maxsize, precision=precision, suppress_small=True)
        # pylint: disable=unnecessary-lambda
        return np.array2string(y, threshold=sys.maxsize, formatter={'all': lambda x: str(x)})

    # pylint: disable=unused-argument
    def to_dataset(self, options: Options, **_) -> Optional[Dataset]:
        """
        See :func:`mlrl.testbed.experiments.output.data.DatasetOutputData.to_dataset`
        """
        return self.dataset
