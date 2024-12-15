"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python code that can be built as wheel packages.
"""
from os import path
from typing import List

from core.modules import Module
from util.files import FileSearch


class PythonPackageModule(Module):
    """
    A module that provides access to Python code that can be built as wheel packages.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonPackageModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonPackageModule)

    def __init__(self, root_directory: str, wheel_directory_name: str):
        """
        :param root_directory:          The path to the module's root directory
        :param wheel_directory_name:    The name of the directory that contains wheel packages
        """
        self.root_directory = root_directory
        self.wheel_directory_name = wheel_directory_name

    @property
    def wheel_directory(self) -> str:
        """
        Returns the path of the directory that contains the wheel packages that have been built for the module.
        """
        return path.join(self.root_directory, self.wheel_directory_name)

    def find_wheels(self) -> List[str]:
        """
        Finds and returns all wheel packages that have been built for the module.

        :return: A list that contains the paths to the wheel packages
        """
        return FileSearch().filter_by_suffix('whl').list(self.wheel_directory)

    def __str__(self) -> str:
        return 'PythonPackageModule {root_directory="' + self.root_directory + '"}'
