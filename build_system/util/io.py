"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing files.
"""
from functools import cached_property
from pathlib import Path
from shutil import rmtree
from typing import Any, List

from util.log import Log

ENCODING_UTF8 = 'utf-8'


def read_file(file: Path):
    """
    Opens a file to read from.

    :param file: The path to the file to be opened
    """
    return open(file, mode='r', encoding=ENCODING_UTF8)


def write_file(file: Path):
    """
    Opens a file to be written to.

    :param file: The path to the file to be opened
    """
    return open(file, mode='w', encoding=ENCODING_UTF8)


def delete_files(*files: Path, accept_missing: bool = True):
    """
    Deletes one or several files or directories.

    :param files:           The files or directories to be deleted
    :param accept_missing:  True, if no error should be raised if the file is missing, False otherwise
    """
    for file in files:
        if file.is_dir():
            Log.verbose('Deleting directory "%s"...', file)
            rmtree(file)
        else:
            if not accept_missing or file.is_file() or file.is_symlink():
                Log.verbose('Deleting file "%s"...', file)
                file.unlink()


def create_directories(*directories: Path):
    """
    Creates one or several directories, if they do not already exist.

    :param directories: The directories to be created
    """
    for directory in directories:
        if not directory.is_dir():
            Log.verbose('Creating directory "%s"...', directory)
            directory.mkdir(parents=True)


class TextFile:
    """
    Allows to read and write the content of a text file.
    """

    def __init__(self, file: Path, accept_missing: bool = False):
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
        if self.accept_missing and not self.file.is_file():
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
        return str(self.file)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.file == other.file

    def __hash__(self) -> int:
        return hash(self.file)
