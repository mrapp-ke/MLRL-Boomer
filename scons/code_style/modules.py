"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to directories and files that belong to individual modules.
"""
from typing import List

from util.files import FileSearch
from util.languages import Language
from util.modules import Module


class CodeModule(Module):
    """
    A module that contains source code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches code modules.
        """

        def __init__(self, *languages: Language):
            """
            :param languages: The languages of the code modules to be matched or None, if no restrictions should be
                              imposed on the languages
            """
            self.languages = set(languages)

        def matches(self, module: Module) -> bool:
            return isinstance(module, CodeModule) and (not self.languages or module.language in self.languages)

    def __init__(self,
                 language: Language,
                 root_directory: str,
                 file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param language:        The programming language of the source code that belongs to the module
        :param root_directory:  The path to the module's root directory
        :param file_search:     The `FileSearch` that should be used to search for source files
        """
        self.language = language
        self.root_directory = root_directory
        self.file_search = file_search

    def find_source_files(self) -> List[str]:
        """
        Finds and returns all source files that belong to the module.

        :return: A list that contains the paths of the source files that have been found
        """
        return self.file_search.set_languages(self.language).list(self.root_directory)
