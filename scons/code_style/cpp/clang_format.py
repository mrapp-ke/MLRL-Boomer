"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "clang-format".
"""
from glob import glob
from os import path

from util.run import Program


class ClangFormat(Program):
    """
    Allows to run the external program "clang-format".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('clang-format', '--style=file')
        self.add_conditional_arguments(enforce_changes, '-i')
        self.add_conditional_arguments(not enforce_changes, '--dry-run', '--Werror')
        self.add_arguments(*glob(path.join(directory, '**', '*.hpp'), recursive=True))
        self.add_arguments(*glob(path.join(directory, '**', '*.cpp'), recursive=True))
