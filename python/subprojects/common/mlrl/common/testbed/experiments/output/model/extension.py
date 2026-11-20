"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write rule learners to HDF5 files.
"""

from pathlib import Path
from typing import override

from mlrl.common.testbed.experiments.output.model.sink_hdf5 import Hdf5FileSink
from mlrl.testbed.experiments.output.model.extension import ModelOutputDirectoryExtension
from mlrl.testbed.experiments.output.sinks import Sink


class Hdf5ModelOutputDirectoryExtension(ModelOutputDirectoryExtension):
    """
    An extension that configures the directory to which rule learners should be written as HDF5 files.
    """

    @override
    def _create_sink(self, directory: Path, create_directory: bool) -> Sink:
        return Hdf5FileSink(directory=directory, create_directory=create_directory)
