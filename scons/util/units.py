"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about independent units of the build system.
"""
from os import path


class BuildUnit:
    """
    An independent unit of the build system that may come with its own built-time dependencies.
    """

    def __init__(self, *subdirectories: str):
        """
        :param subdirectories: The subdirectories within the build system that lead to the root directory of this unit
        """
        self.root_directory = path.join('scons', *subdirectories)

    @property
    def requirements_file(self) -> str:
        """
        The path to the requirements file that specifies the build-time dependencies of this unit.
        """
        return path.join(self.root_directory, 'requirements.txt')
