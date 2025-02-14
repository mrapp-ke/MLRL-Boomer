"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing version files.
"""
from util.io import TextFile


class PythonVersionFile(TextFile):
    """
    A file that stores the minimum Python version required by the project.
    """

    def __init__(self):
        super().__init__('.version-python')

    @property
    def version(self) -> str:
        """
        The version that is stored in the file.
        """
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + self.file + '" must contain exactly one line')

        return lines[0]
