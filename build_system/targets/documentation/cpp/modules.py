"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to C++ code for which an API documentation can be generated.
"""
from os import environ, path
from typing import List

from core.modules import Module, SubprojectModule
from util.files import FileSearch, FileType

from targets.documentation.modules import ApidocModule


class CppApidocModule(ApidocModule):
    """
    A module that provides access to C++ code for which an API documentation can be generated.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `CppApidocModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, CppApidocModule) and SubprojectModule.Filter.from_env(environ).matches(module)

    def __init__(self,
                 root_directory: str,
                 output_directory: str,
                 include_directory_name: str,
                 header_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param output_directory:        The path to the directory where the API documentation should be stored
        :param include_directory_name:  The name of the directory that contains the header files to be included in the
                                        API documentation
        :param header_file_search:      The `FileSearch` that should be used to search for the header files to be
                                        included in the API documentation
        """
        super().__init__(output_directory)
        self.root_directory = root_directory
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

    @property
    def subproject_name(self) -> str:
        return path.basename(self.root_directory)

    def create_reference(self) -> str:
        return 'Library libmlrl' + self.subproject_name + ' <' + path.join(path.basename(self.output_directory),
                                                                           'filelist.rst') + '>'

    def __str__(self) -> str:
        return 'CppApidocModule {root_directory="' + self.root_directory + '"}'
