"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
from os import path
from typing import Optional

# The suffix of an XML file
SUFFIX_XML = 'xml'

# UTF-8 encoding
ENCODING_UTF8 = 'utf-8'


def get_file_name(name: str, suffix: str) -> str:
    """
    Returns a file name, including a suffix.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :return:        The file name
    """
    return name + '.' + suffix


def get_file_name_per_fold(name: str, suffix: str, fold: Optional[int]) -> str:
    """
    Returns a file name, including a suffix, that corresponds to a certain fold.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :param fold:    The cross validation fold, the file corresponds to, or None, if the file does not correspond to a
                    specific fold
    :return:        The file name
    """
    return get_file_name(name + '_' + ('overall' if fold is None else 'fold-' + str(fold + 1)), suffix)


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
