"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to directories and files that belong to individual modules.
"""
from typing import List

from util.files import FileSearch, FileType
from util.modules import Module


class CodeModule(Module):
    """
    A module that contains source code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches code modules.
        """

        def __init__(self, *file_types: FileType):
            """
            :param file_types: The file types of the code modules to be matched or None, if no restrictions should be
                               imposed on the file types
            """
            self.file_types = set(file_types)

        def matches(self, module: Module) -> bool:
            return isinstance(module, CodeModule) and (not self.file_types or module.file_type in self.file_types)

    def __init__(self,
                 file_type: FileType,
                 root_directory: str,
                 source_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param file_type:           The `FileType` of the source files that belongs to the module
        :param root_directory:      The path to the module's root directory
        :param source_file_search:  The `FileSearch` that should be used to search for source files
        """
        self.file_type = file_type
        self.root_directory = root_directory
        self.source_file_search = source_file_search

    def find_source_files(self) -> List[str]:
        """
        Finds and returns all source files that belong to the module.

        :return: A list that contains the paths of the source files that have been found
        """
        return self.source_file_search.filter_by_file_type(self.file_type).list(self.root_directory)
