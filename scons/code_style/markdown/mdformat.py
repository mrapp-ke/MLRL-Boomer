"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "mdformat".
"""
from glob import glob
from os import path

from util.run import Program


class MdFormat(Program):
    """
    Allows to run the external program "mdformat".
    """

    def __init__(self, directory: str, recursive: bool = False, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param recursive:       True, if the program should be applied to subdirectories, False otherwise
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('mdformat', '--number', '--wrap', 'no', '--end-of-line', 'lf')
        self.add_conditional_arguments(not enforce_changes, '--check')
        suffix_md = '*.md'
        glob_path = path.join(directory, '**', '**', suffix_md) if recursive else path.join(directory, suffix_md)
        self.add_arguments(*glob(glob_path, recursive=recursive))
        self.add_dependencies('mdformat-myst')
