"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to a Sphinx documentation.
"""
from abc import ABC, abstractmethod
from typing import List

from core.modules import Module
from util.files import FileSearch


class ApidocModule(Module, ABC):
    """
    An abstract base class for all modules that provide access to source code for which an API documentation can be
    generated.
    """

    def __init__(self, output_directory: str):
        """
        :param output_directory: The path to the directory where the API documentation should be stored
        """
        self.output_directory = output_directory

    @abstractmethod
    def create_reference(self) -> str:
        """
        Must be implemented by subclasses in order to create a reference to API documentation.

        :return: The reference that has been created
        """


class SphinxModule(Module):
    """
    A module that provides access to a Sphinx documentation.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `SphinxModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, SphinxModule)

    def __init__(self,
                 root_directory: str,
                 output_directory: str,
                 source_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:      The path to the module's root directory
        :param output_directory:    The path to the directory where the documentation should be stored
        :param source_file_search:  The `FileSearch` that should be used to search for the source files of the
                                    documentation
        """
        self.root_directory = root_directory
        self.output_directory = output_directory
        self.source_file_search = source_file_search

    def find_source_files(self) -> List[str]:
        """
        Finds and returns all source files of the documentation.

        :return: A list that contains the source files that have been found
        """
        return self.source_file_search.list(self.root_directory)

    def __str__(self) -> str:
        return 'SphinxModule {root_directory="' + self.root_directory + '"}'
