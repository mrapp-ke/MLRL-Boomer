"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
from functools import cached_property
from typing import List

ENCODING_UTF8 = 'utf-8'


def read_file(file: str):
    """
    Opens a file to read from.

    :param file: The file to be opened
    """
    return open(file, mode='r', encoding=ENCODING_UTF8)


def write_file(file: str):
    """
    Opens a file to be written to.

    :param file: The file to be opened
    """
    return open(file, mode='w', encoding=ENCODING_UTF8)


class TextFile:
    """
    Allows to read and write the content of a text file.
    """

    def __init__(self, file: str):
        """
        :param file: The path to the text file
        """
        self.file = file

    @cached_property
    def lines(self) -> List[str]:
        """
        The lines in the text file.
        """
        with read_file(self.file) as file:
            return file.readlines()

    def write_lines(self, lines: List[str]):
        """
        Overwrites all lines in the text file.

        :param lines: The lines to be written
        """
        with write_file(self.file) as file:
            file.writelines(lines)

        del self.lines
