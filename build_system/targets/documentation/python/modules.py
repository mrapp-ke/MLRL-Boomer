"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python code for which an API documentation can be generated.
"""
from os import environ, path
from typing import List

from core.modules import Module, SubprojectModule
from util.files import FileSearch, FileType

from targets.documentation.modules import ApidocModule


class PythonApidocModule(ApidocModule):
    """
    A module that provides access to Python code for which an API documentation can be generated.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonApidocModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonApidocModule) and SubprojectModule.Filter.from_env(environ).matches(module)

    def __init__(self,
                 root_directory: str,
                 output_directory: str,
                 source_directory_name: str,
                 source_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param output_directory:        The path to the directory where the API documentation should be stored
        :param source_directory_name:   The name of the directory that contains the Python source files to be included
                                        in the API documentation
        :param source_file_search:      The `FileSearch` that should be used to search for the header files to be
                                        included in the API documentation
        """
        super().__init__(output_directory)
        self.root_directory = root_directory
        self.source_directory_name = source_directory_name
        self.source_file_search = source_file_search

    @property
    def source_directory(self) -> str:
        """
        The path to the directory that contains the Python source files to be included in the API documentation.
        """
        return path.join(self.root_directory, self.source_directory_name)

    def find_source_files(self) -> List[str]:
        """
        Finds and returns the Python source files to be included in the API documentation.

        :return: A list that contains the source files that have been found
        """
        return self.source_file_search.filter_by_file_type(FileType.python()).list(self.source_directory)

    @property
    def subproject_name(self) -> str:
        return path.basename(self.output_directory)

    def create_reference(self) -> str:
        return 'Package mlrl-' + path.basename(self.output_directory) + ' <' + path.join(
            self.subproject_name, self.source_directory_name + '.' + self.subproject_name + '.rst') + '>'

    def __str__(self) -> str:
        return 'PythonApidocModule {root_directory="' + self.root_directory + '"}'
