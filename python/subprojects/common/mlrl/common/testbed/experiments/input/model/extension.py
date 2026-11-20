"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read rule learners from HDF5 files.
"""

from pathlib import Path
from typing import override

from mlrl.common.testbed.experiments.input.model.source_hdf5 import Hdf5FileSource
from mlrl.testbed.experiments.input.model.extension import ModelInputExtension
from mlrl.testbed.experiments.input.sources.source import Source


class Hdf5ModelInputExtension(ModelInputExtension):
    """
    An extension that configures the functionality to read rule learners from HDF5 files.
    """

    @override
    def _create_source(self, directory: Path) -> Source:
        return Hdf5FileSource(directory)
