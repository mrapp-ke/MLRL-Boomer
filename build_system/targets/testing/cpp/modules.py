"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests for C++ code.
"""
from os import path
from typing import override

from core.modules import Module, ModuleRegistry

from targets.testing.modules import TestModule


class CppTestModule(TestModule):
    """
    A module that provides access to automated tests for C++ code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `CppTestModule`.
        """

        @override
        def matches(self, module: Module, _: ModuleRegistry) -> bool:
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

    @override
    def __str__(self) -> str:
        return 'CppTestModule {root_directory="' + self.root_directory + '"}'
