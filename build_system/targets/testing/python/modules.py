"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests for Python code.
"""
from core.modules import Module

from targets.testing.modules import TestModule


class PythonTestModule(TestModule):
    """
    A module that provides access to automated tests for Python code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonTestModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonTestModule)

    def __init__(self, root_directory: str, result_directory: str):
        """
        :param root_directory:      The path to the module's root directory
        :param result_directory:    The path to the directory, where test results should be stored
        """
        self.root_directory = root_directory
        self.result_directory = result_directory

    def __str__(self) -> str:
        return 'PythonTestModule {root_directory="' + self.root_directory + '"}'
