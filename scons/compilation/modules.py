"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to directories and files that belong to individual modules.
"""
from os import path
from typing import List, Optional

from util.files import FileSearch, FileType
from util.modules import Module


class CompilationModule(Module):
    """
    A module that contains source code that must be compiled.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules that contain source code that must be compiled.
        """

        def __init__(self, *file_types: FileType):
            """
            :param file_types: The file types of the source files contained by the modules to be matched or None, if no
                               restrictions should be imposed on the file types
            """
            self.file_types = set(file_types)

        def matches(self, module: Module) -> bool:
            return isinstance(module, CompilationModule) and (not self.file_types
                                                              or module.file_type in self.file_types)

    def __init__(self,
                 file_type: FileType,
                 root_directory: str,
                 build_directory_name: str,
                 install_directory: Optional[str] = None,
                 installed_file_search: Optional[FileSearch] = None):
        """
        :param file_type:               The file types of the source files that belongs to the module
        :param root_directory:          The path to the module's root directory
        :param build_directory_name:    The name of the module's build directory
        :param install_directory:       The path to the directory into which files are installed or None, if the files
                                        are installed into the root directory
        :param installed_file_search:   The `FileSearch` that should be used to search for installed files or None, if
                                        the module does never contain any installed files
        """
        self.file_type = file_type
        self.root_directory = root_directory
        self.build_directory_name = build_directory_name
        self.install_directory = install_directory if install_directory else root_directory
        self.installed_file_search = installed_file_search

    @property
    def build_directory(self) -> str:
        """
        The path to the directory, where build files should be stored.
        """
        return path.join(self.root_directory, self.build_directory_name)

    def find_installed_files(self) -> List[str]:
        """
        Finds and returns all installed files that belong to the module.

        :return: A list that contains the paths of the requirements files that have been found
        """
        if self.installed_file_search:
            return self.installed_file_search \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(path.basename(self.build_directory)) \
                .list(self.install_directory)

        return []
