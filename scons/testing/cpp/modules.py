"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to automated tests for C++ code that belong to individual modules.
"""
from os import path

from testing.modules import TestModule
from util.modules import Module


class CppTestModule(TestModule):
    """
    A module that contains automated tests for C++ code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules that contain automated tests for C++ code.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, CppTestModule)

    def __init__(self, root_directory: str, build_directory_name: str):
        """
        :param root_directory:          The path to the module's root directory
        :param build_directory_name:    The name of the module's build directory
        """
        self.root_directory = root_directory
        self.build_directory_name = build_directory_name

    @property
    def build_directory(self) -> str:
        """
        The path to the directory, where build files are stored.
        """
        return path.join(self.root_directory, self.build_directory_name)
