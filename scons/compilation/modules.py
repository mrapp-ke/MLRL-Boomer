"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to directories and files that belong to individual modules.
"""
from os import path

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

    def __init__(self, language: Language, root_directory: str):
        """
        :param language:        The programming language of the source code that belongs to the module
        :param root_directory:  The path to the module's root directory
        """
        self.language = language
        self.root_directory = root_directory

    @property
    def build_directory(self) -> str:
        """
        The path to the directory, where build files should be stored.
        """
        return path.join(self.root_directory, 'build')
