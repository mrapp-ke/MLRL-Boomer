"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "yapf".
"""
from util.run import Program


class Yapf(Program):
    """
    Allows to run the external program "yapf".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yapf', '-r', '-p', '--style=.style.yapf', '--exclude', '**/build/*.py',
                         '-i' if enforce_changes else '--diff', directory)
