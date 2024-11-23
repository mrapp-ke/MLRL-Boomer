"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "isort".
"""
from util.run import Program


class ISort(Program):
    """
    Allows to run the external program "isort".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('isort', directory, '--settings-path', '.', '--virtual-env', 'venv', '--skip-gitignore')
        self.add_conditional_arguments(not enforce_changes, '--check')
