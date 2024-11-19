"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""

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
