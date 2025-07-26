"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about independent units of the build system.
"""
from pathlib import Path
from typing import List


class BuildUnit:
    """
    An independent unit of the build system that may come with its own built-time dependencies.
    """

    BUILD_SYSTEM_DIRECTORY = Path('build_system')

    BUILD_DIRECTORY_NAME = 'build'

    def __init__(self, root_directory: Path = BUILD_SYSTEM_DIRECTORY):
        """
        :param root_directory: The root directory of this unit
        """
        self.root_directory = root_directory

    @staticmethod
    def for_file(file: Path) -> 'BuildUnit':
        """
        Creates and returns a `BuildUnit` for a given file.

        :param file:    The path to the file for which a `BuildUnit` should be created
        :return:        The `BuildUnit` that has been created
        """
        return BuildUnit(file.parent)

    @property
    def build_directory(self) -> Path:
        """
        The path to the build directory of this unit.
        """
        return self.root_directory / self.BUILD_DIRECTORY_NAME

    def find_requirements_files(self) -> List[Path]:
        """
        A list that contains the paths to all requirements files that specify the build-time dependencies of this unit.
        """
        requirements_files = []
        current_directory = self.root_directory

        while not current_directory.samefile(self.BUILD_SYSTEM_DIRECTORY):
            requirements_file = current_directory / 'requirements.txt'

            if requirements_file.is_file():
                requirements_files.append(requirements_file)

            current_directory = current_directory.parent

        return requirements_files
