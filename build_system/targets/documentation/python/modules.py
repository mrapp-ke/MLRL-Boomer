"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python code for which an API documentation can be generated.
"""
from os import environ
from pathlib import Path
from typing import List, override

from core.modules import Module, ModuleRegistry
from util.files import FileSearch, FileType

from targets.documentation.modules import ApidocModule
from targets.modules import SubprojectModule


class PythonApidocModule(ApidocModule):
    """
    A module that provides access to Python code for which an API documentation can be generated.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonApidocModule`.
        """

        @override
        def matches(self, module: Module, module_registry: ModuleRegistry) -> bool:
            return isinstance(module, PythonApidocModule) and SubprojectModule.Filter.from_env(environ).matches(
                module, module_registry)

    def __init__(self,
                 root_directory: Path,
                 output_directory: Path,
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
    def source_directory(self) -> Path:
        """
        The path to the directory that contains the Python source files to be included in the API documentation.
        """
        return self.root_directory / self.source_directory_name

    def find_source_files(self) -> List[Path]:
        """
        Finds and returns the Python source files to be included in the API documentation.

        :return: A list that contains the source files that have been found
        """
        return self.source_file_search.filter_by_file_type(FileType.python()).list(self.source_directory)

    @override
    @property
    def subproject_name(self) -> str:
        return self.output_directory.name

    @override
    def create_reference(self) -> str:
        rst_file_name = self.source_directory_name + '.' + self.subproject_name.replace('-', '_') + '.rst'
        rst_file_path = Path(self.subproject_name, rst_file_name)
        return 'Package mlrl-' + self.output_directory.name + ' <' + str(rst_file_path) + '>'

    @override
    def __str__(self) -> str:
        return 'PythonApidocModule {root_directory="' + str(self.root_directory) + '"}'
