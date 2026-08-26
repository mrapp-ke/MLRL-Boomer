"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing tabular datasets that are part of output data.
"""

import sys
from typing import override

import numpy as np
from rich.console import ConsoleRenderable

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties
from mlrl.testbed.experiments.output.dataset.dataset import OutputDataset
from mlrl.testbed.util.format import OPTION_DECIMALS
from mlrl.testbed_sklearn.experiments.dataset import TabularDataset
from mlrl.util.options import Options


class TabularOutputDataset(OutputDataset):
    """
    Represents a tabular dataset that is part of output data.
    """

    def __init__(self, dataset: TabularDataset, properties: Properties, context: Context | None = None):
        """
        :param dataset:     A tabular dataset
        :param properties:  The properties of the output data
        :param context:     A `Context` to be used by default for finding a suitable sink this output data can be
                            written to or None, if the default should be used
        """
        super().__init__(dataset=dataset.enforce_dense_outputs(), properties=properties, context=context)

    @override
    def to_text(self, options: Options, **_) -> str | ConsoleRenderable | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.TextualOutputData.to_text`
        """
        y = self.dataset.enforce_dense_outputs().y

        if y.dtype.kind == 'f':
            decimals = options.get_int(OPTION_DECIMALS, 2)
            precision = decimals if decimals > 0 else None
            return np.array2string(y, threshold=sys.maxsize, precision=precision, suppress_small=True)
        return np.array2string(y, threshold=sys.maxsize, formatter={'all': lambda x: str(x)})
