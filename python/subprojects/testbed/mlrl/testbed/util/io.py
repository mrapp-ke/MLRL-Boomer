"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
from os import path

ENCODING_UTF8 = 'utf-8'


def open_readable_file(file_path: str):
    """
    Opens a file to be read from.

    :param file_path:   The path to the file to be opened
    :return:            The file that has been opened
    """
    return open(file_path, mode='r', newline='', encoding=ENCODING_UTF8)


def open_writable_file(file_path: str, append: bool = False):
    """
    Opens a file to be written to.

    :param file_path:   The path to the file to be opened
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    mode = 'a' if append and path.isfile(file_path) else 'w'
    return open(file_path, mode=mode, encoding=ENCODING_UTF8)
