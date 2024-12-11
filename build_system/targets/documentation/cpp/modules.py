"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to C++ code for which an API documentation can be generated.
"""
from os import path
from typing import List

from core.modules import Module
from util.files import FileSearch, FileType


class CppApidocModule(Module):
    """
    A module that contains C++ code for which an API documentation can be generated.
    """

    class Filter(Module.Filter):
        """
        A filter that matches code modules.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, CppApidocModule)

    def __init__(self,
                 root_directory: str,
                 output_directory: str,
                 project_name: str,
                 include_directory_name: str,
                 header_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param output_directory:        The path to the directory where the API documentation should be stored
        :param project_name:            The name of the C++ project to be documented
        :param include_directory_name:  The name of the directory that contains the header files to be included in the
                                        API documentation
        :param header_file_search:      The `FileSearch` that should be used to search for the header files to be
                                        included in the API documentation
        """
        self.root_directory = root_directory
        self.output_directory = output_directory
        self.project_name = project_name
        self.include_directory_name = include_directory_name
        self.header_file_search = header_file_search

    @property
    def include_directory(self) -> str:
        """
        The path to the directory that contains the header files to be included in the API documentation.
        """
        return path.join(self.root_directory, self.include_directory_name)

    def find_header_files(self) -> List[str]:
        """
        Finds and returns the header files to be included in the API documentation.

        :return: A list that contains the header files that have been found
        """
        return self.header_file_search.filter_by_file_type(FileType.cpp()).list(self.include_directory)

    def __str__(self) -> str:
        return 'CppApidocModule {root_directory="' + self.root_directory + '"}'
