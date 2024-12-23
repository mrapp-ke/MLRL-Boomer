"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python requirements files.
"""
from enum import Enum
from os import environ, path
from typing import List

from core.modules import Module
from util.files import FileSearch

from targets.modules import SubprojectModule


class DependencyType(Enum):
    """
    The type of the Python dependencies.
    """
    BUILD_TIME = 'build-time'
    RUNTIME = 'runtime'


class PythonDependencyModule(SubprojectModule):
    """
    A module that provides access to Python requirements files.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonDependencyModule`.
        """

        def __init__(self, *dependency_types: DependencyType):
            """
            :param dependency_types: The type of the Python dependencies of the modules to be matched or None, if no
                                     restrictions should be imposed on the types of dependencies
            """
            self.dependency_types = set(dependency_types)

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonDependencyModule) and (
                not self.dependency_types or module.dependency_type
                in self.dependency_types) and SubprojectModule.Filter.from_env(environ).matches(module)

    def __init__(self,
                 dependency_type: DependencyType,
                 root_directory: str,
                 requirements_file_search: FileSearch = FileSearch()):
        """
        :param dependency_type:             The type of the Python dependencies
        :param root_directory:              The path to the module's root directory
        :param requirements_file_search:    The `FileSearch` that should be used to search for requirements files
        """
        self.dependency_type = dependency_type
        self.root_directory = root_directory
        self.requirements_file_search = requirements_file_search

    def find_requirements_files(self) -> List[str]:
        """
        Finds and returns all requirements files that belong to the module.

        :return: A list that contains the paths of the requirements files that have been found
        """
        return self.requirements_file_search.filter_by_name('requirements.txt').list(self.root_directory)

    @property
    def subproject_name(self) -> str:
        return path.basename(self.root_directory)

    def __str__(self) -> str:
        return ('PythonDependencyModule {dependency_type="' + self.dependency_type.value + '", root_directory="'
                + self.root_directory + '"}')
