"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading input data from files using Python's pickle mechanism.
"""
import logging as log
import pickle

from pathlib import Path
from typing import Any, Optional, override

from mlrl.testbed.experiments.input.data import InputData
from mlrl.testbed.experiments.input.sources.source import FileSource
from mlrl.testbed.experiments.output.sinks.sink_pickle import PickleFileSink
from mlrl.testbed.experiments.state import ExperimentState


class PickleFileSource(FileSource):
    """
    Allows to read input data from a file using Python's pickle mechanism.
    """

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=PickleFileSink.SUFFIX_PICKLE)

    @override
    def _read_from_file(self, state: ExperimentState, file_path: Path, input_data: InputData) -> Optional[Any]:
        try:
            with open(file_path, mode='rb') as pickle_file:
                return pickle.load(pickle_file)
        except IOError:
            log.error('Failed to unpickle file \"%s\"', file_path)
            return None
