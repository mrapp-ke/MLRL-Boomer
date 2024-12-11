"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about independent units of the build system.
"""
from os import path
from typing import List


class BuildUnit:
    """
    An independent unit of the build system that may come with its own built-time dependencies.
    """

    BUILD_SYSTEM_DIRECTORY = 'build_system'

    BUILD_DIRECTORY_NAME = 'build'

    def __init__(self, root_directory: str = BUILD_SYSTEM_DIRECTORY):
        """
        :param root_directory: The root directory of this unit
        """
        self.root_directory = root_directory

    @staticmethod
    def for_file(file) -> 'BuildUnit':
        """
        Creates and returns a `BuildUnit` for a given file.

        :param file:    The file for which a `BuildUnit` should be created
        :return:        The `BuildUnit` that has been created
        """
        return BuildUnit(path.relpath(path.dirname(file), path.dirname(BuildUnit.BUILD_SYSTEM_DIRECTORY)))

    @property
    def build_directory(self) -> str:
        """
        The path to the build directory of this unit.
        """
        return path.join(self.root_directory, self.BUILD_DIRECTORY_NAME)

    def find_requirements_files(self) -> List[str]:
        """
        The path to the requirements file that specifies the build-time dependencies of this unit.
        """
        requirements_files = []
        current_directory = self.root_directory

        while path.basename(current_directory) != self.BUILD_SYSTEM_DIRECTORY:
            requirements_file = path.join(current_directory, 'requirements.txt')

            if path.isfile(requirements_file):
                requirements_files.append(requirements_file)

            current_directory = path.dirname(current_directory)

        return requirements_files
