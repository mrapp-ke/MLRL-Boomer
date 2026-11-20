"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing rule learners to HDF5 files.
"""

from pathlib import Path
from typing import override

from mlrl.common.learners import RuleLearner
from mlrl.common.testbed.experiments.input.model.source_hdf5 import Hdf5FileSource
from mlrl.testbed.experiments.input.sources import Source
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.model.model import OutputModel
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.util.options import Options


class Hdf5FileSink(FileSink):
    """
    Allows to write rule learners to HDF5 files by using the function `RuleLearner.save_model`.
    """

    def __init__(self, directory: Path, options: Options | None = None, create_directory: bool = False):
        """
        :param directory:           The path to the directory of the file
        :param options:             Options to be taken into account or None, if the defaults should be used
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(
            directory=directory,
            suffix=Hdf5FileSource.SUFFIX_HDF5,
            options=options,
            create_directory=create_directory,
        )

    @override
    def _write_to_file(self, file_path: Path, state: ExperimentState, output_data: OutputData, **_):
        if isinstance(output_data, OutputModel):
            learner = output_data.to_object(self.options)

            if learner is not None:
                if not isinstance(learner, RuleLearner):
                    raise TypeError(
                        f'Only rule learners can be saved as HDF5 files, but got an object of type '
                        f'{type(learner).__name__}'
                    )

                learner.save_model(file_path)

    @override
    def create_source(self, input_directory: Path) -> Source | None:
        return Hdf5FileSource(input_directory)
