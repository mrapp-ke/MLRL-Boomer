"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python requirements files.
"""
from enum import Enum
from os import path
from typing import List

from core.build_unit import BuildUnit
from core.modules import Module
from util.files import FileSearch
from util.pip import RequirementsFile, RequirementsTextFile

from targets.dependencies.python.pyproject_toml_file import PyprojectTomlFile
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

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonDependencyModule)

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

    def find_requirements_files(self, build_unit: BuildUnit, dependency_type: DependencyType) -> List[RequirementsFile]:
        """
        Finds and returns all requirements files that belong to the module and match a given `DependencyType`.

        :param build_unit:      The `BuildUnit` from which this method is invoked
        :param dependency_type: The `DependencyType` to be matched
        :return:                A list that contains the requirements files that have been found
        """
        requirements_files = []

        if self.dependency_type == dependency_type:
            requirements_files.extend([
                RequirementsTextFile(file)
                for file in self.requirements_file_search.filter_by_name('requirements.txt').list(self.root_directory)
            ])

        if dependency_type == DependencyType.BUILD_TIME:
            pyproject_toml_file = path.join(self.root_directory, 'pyproject.template.toml')

            if path.isfile(pyproject_toml_file):
                requirements_files.append(PyprojectTomlFile(build_unit, pyproject_toml_file))

        return requirements_files

    @property
    def subproject_name(self) -> str:
        return path.basename(self.root_directory)

    def __str__(self) -> str:
        return ('PythonDependencyModule {dependency_type="' + self.dependency_type.value + '", root_directory="'
                + self.root_directory + '"}')
