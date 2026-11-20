"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading rule learners from HDF5 files.
"""

from pathlib import Path
from typing import Any, override

from mlrl.common.learners import load_model
from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.input.model.model import InputModel
from mlrl.testbed.experiments.input.sources.source import FileSource
from mlrl.testbed.experiments.state import ExperimentState


class Hdf5FileSource(FileSource):
    """
    Allows to read rule learners from HDF5 files that have been written via the function `RuleLearner.save_model`.
    """

    SUFFIX_HDF5 = 'h5'

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_HDF5)

    @override
    def _read_from_file(self, state: ExperimentState, file_path: Path, input_data: InputData) -> Any | None:
        if isinstance(input_data, InputModel):
            return load_model(file_path)

        return None
