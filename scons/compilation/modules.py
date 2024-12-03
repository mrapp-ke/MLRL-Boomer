"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to directories and files that belong to individual modules.
"""
from os import path
from typing import List, Optional

from util.files import FileSearch
from util.languages import Language
from util.modules import Module


class CompilationModule(Module):
    """
    A module that contains source code that must be compiled.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules that contain source code that must be compiled.
        """

        def __init__(self, *languages: Language):
            """
            :param languages: The languages of the source code contained by the modules to be matched or None, if no
                              restrictions should be imposed on the languages
            """
            self.languages = set(languages)

        def matches(self, module: Module) -> bool:
            return isinstance(module, CompilationModule) and (not self.languages or module.language in self.languages)

    def __init__(self,
                 language: Language,
                 root_directory: str,
                 install_directory: Optional[str] = None,
                 installed_file_search: Optional[FileSearch] = None):
        """
        :param language:                The programming language of the source code that belongs to the module
        :param root_directory:          The path to the module's root directory
        :param install_directory:       The path to the directory into which files are installed or None, if the files
                                        are installed into the root directory
        :param installed_file_search:   The `FileSearch` that should be used to search for installed files or None, if
                                        the module does never contain any installed files
        """
        self.language = language
        self.root_directory = root_directory
        self.install_directory = install_directory if install_directory else root_directory
        self.installed_file_search = installed_file_search

    @property
    def build_directory(self) -> str:
        """
        The path to the directory, where build files should be stored.
        """
        return path.join(self.root_directory, 'build')

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
