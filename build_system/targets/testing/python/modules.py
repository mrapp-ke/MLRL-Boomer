"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests for Python code.
"""
from os import environ
from pathlib import Path
from typing import override

from core.modules import Module, ModuleRegistry

from targets.modules import SubprojectModule
from targets.testing.modules import TestModule


class PythonTestModule(TestModule, SubprojectModule):
    """
    A module that provides access to automated tests for Python code.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonTestModule`.
        """

        @override
        def matches(self, module: Module, module_registry: ModuleRegistry) -> bool:
            return isinstance(module, PythonTestModule) and SubprojectModule.Filter.from_env(environ).matches(
                module, module_registry)

    def __init__(self, root_directory: Path, result_directory: Path):
        """
        :param root_directory:      The path to the module's root directory
        :param result_directory:    The path to the directory, where test results should be stored
        """
        self.root_directory = root_directory
        self.result_directory = result_directory

    @override
    @property
    def subproject_name(self) -> str:
        return self.root_directory.name.replace('_', '-')

    @override
    def __str__(self) -> str:
        return 'PythonTestModule {root_directory="' + str(self.root_directory) + '"}'
