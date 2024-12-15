"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests for Python code.
"""
from os import path
from typing import List

from core.modules import Module
from util.files import FileSearch, FileType

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

    def __init__(self,
                 root_directory: str,
                 build_directory_name: str,
                 test_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param build_directory_name:    The name of the module's build directory
        :param test_file_search:        The `FilesSearch` that should be used to search for test files
        """
        self.root_directory = root_directory
        self.build_directory_name = build_directory_name
        self.test_file_search = test_file_search

    @property
    def test_result_directory(self) -> str:
        """
        The path of the directory where tests results should be stored.
        """
        return path.join(self.root_directory, self.build_directory_name, 'test-results')

    def find_test_directories(self) -> List[str]:
        """
        Finds and returns all directories that contain automated tests that belong to the module.

        :return: A list that contains the paths of the directories that have been found
        """
        test_files = self.test_file_search \
            .exclude_subdirectories_by_name(self.build_directory_name) \
            .filter_by_substrings(starts_with='test_') \
            .filter_by_file_type(FileType.python()) \
            .list(self.root_directory)
        return list({path.dirname(test_file) for test_file in test_files})

    def __str__(self) -> str:
        return 'PythonTestModule {root_directory="' + self.root_directory + '"}'
