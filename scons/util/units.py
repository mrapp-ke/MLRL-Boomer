"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about independent units of the build system.
"""
from dataclasses import dataclass
from os import path


@dataclass
class BuildUnit:
    """
    An independent unit of the build system that may come with its own built-time dependencies.

    Attributes:
        root_directory:     The path to the root directory of this unit
        requirements_file:  The path to the requirements file that specifies the dependencies required by this unit
    """

    def __init__(self, root_directory: str = path.join('scons', 'util')):
        self.root_directory = root_directory
        self.requirements_file: str = path.join(root_directory, 'requirements.txt')

    @staticmethod
    def by_name(unit_name: str, *subdirectories: str) -> 'BuildUnit':
        """
        Creates and returns a `BuildUnit` with a specific name.

        :param unit_name:       The name of the build unit
        :param subdirectories:  Optional subdirectories
        :return:                The `BuildUnit` that has been created
        """
        return BuildUnit(path.join('scons', unit_name, *subdirectories))
