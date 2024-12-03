"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to automated tests for Python code that belong to individual modules.
"""
from os import path
from typing import List

from testing.modules import TestModule
from util.files import DirectorySearch
from util.modules import Module


class PythonTestModule(TestModule):
    """
    A module that contains automated tests for Python code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules that contain automated tests for Python code.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonTestModule)

    def __init__(self,
                 root_directory: str,
                 directory_search: DirectorySearch = DirectorySearch().set_recursive(True).exclude_by_name(
                     'build').filter_by_name('tests')):
        """
        :param root_directory:      The path to the module's root directory
        :param directory_search:    The `DirectorySearch` that should be used for directories containing automated tests
        """
        self.root_directory = root_directory
        self.directory_search = directory_search

    @property
    def test_result_directory(self) -> str:
        """
        The path of the directory where tests results should be stored.
        """
        return path.join(self.root_directory, 'build', 'test-results')

    def find_test_directories(self) -> List[str]:
        """
        Finds and returns all directories that contain automated tests that belong to the module.

        :return: A list that contains the paths of the directories that have been found
        """
        return self.directory_search.list(self.root_directory)
