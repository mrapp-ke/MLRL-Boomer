"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about independent units of the build system.
"""
from os import path
from typing import List

from util.paths import Project


class BuildUnit:
    """
    An independent unit of the build system that may come with its own built-time dependencies.
    """

    def __init__(self, *subdirectories: str):
        """
        :param subdirectories: The subdirectories within the build system that lead to the root directory of this unit
        """
        self.root_directory = path.join(Project.BuildSystem.root_directory, *subdirectories)

    def find_requirements_files(self) -> List[str]:
        """
        The path to the requirements file that specifies the build-time dependencies of this unit.
        """
        requirements_files = []
        current_directory = self.root_directory

        while path.basename(current_directory) != Project.BuildSystem.root_directory:
            requirements_file = path.join(current_directory, 'requirements.txt')

            if path.isfile(requirements_file):
                requirements_files.append(requirements_file)

            current_directory = path.dirname(current_directory)

        return requirements_files
