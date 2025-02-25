"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
from functools import cached_property
from os import makedirs, path, remove
from shutil import rmtree
from typing import List

from util.log import Log

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


def delete_files(*files: str, accept_missing: bool = True):
    """
    Deletes one or several files or directories.

    :param files:           The files or directories to be deleted
    :param accept_missing:  True, if no error should be raised if the file is missing, False otherwise
    """
    for file in files:
        if path.isdir(file):
            Log.verbose('Deleting directory "%s"...', file)
            rmtree(file)
        else:
            if not accept_missing or path.isfile(file) or path.islink(file):
                Log.verbose('Deleting file "%s"...', file)
                remove(file)


def create_directories(*directories: str):
    """
    Creates one or several directories, if they do not already exist.

    :param directories: The directories to be created
    """
    for directory in directories:
        if not path.isdir(directory):
            Log.verbose('Creating directory "%s"...', directory)
            makedirs(directory)


class TextFile:
    """
    Allows to read and write the content of a text file.
    """

    def __init__(self, file: str, accept_missing: bool = False):
        """
        :param file:            The path to the text file
        :param accept_missing:  True, if no errors should be raised if the text file is missing, False otherwise
        """
        self.file = file
        self.accept_missing = accept_missing

    @cached_property
    def lines(self) -> List[str]:
        """
        The lines in the text file.
        """
        if self.accept_missing and not path.isfile(self.file):
            return []

        with read_file(self.file) as file:
            return file.readlines()

    def write_lines(self, *lines: str):
        """
        Overwrites all lines in the text file.

        :param lines: The lines to be written
        """
        with write_file(self.file) as file:
            file.writelines(lines)

        try:
            del self.lines
        except AttributeError:
            pass

    def clear(self):
        """
        Clears the text file.
        """
        Log.info('Clearing file "%s"...', self.file)
        self.write_lines('')

    def delete(self):
        """
        Deletes the text file.
        """
        delete_files(self.file, accept_missing=self.accept_missing)

    def __str__(self) -> str:
        return self.file

    def __eq__(self, other: 'TextFile') -> bool:
        return self.file == other.file

    def __hash__(self) -> int:
        return hash(self.file)
