"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading textual input data from a text file.
"""

from pathlib import Path
from typing import Optional, override

from mlrl.testbed.experiments.input.data import TextualInputData
from mlrl.testbed.experiments.input.sources.source import TextualFileSource
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import open_readable_file


class TextFileSource(TextualFileSource):
    """
    Allows to read textual input data from a text file.
    """

    SUFFIX_TEXT = 'txt'

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_TEXT)

    @override
    def _read_text_from_file(self, _: ExperimentState, file_path: Path, input_data: TextualInputData) -> Optional[str]:
        with open_readable_file(file_path) as text_file:
            return text_file.read()
