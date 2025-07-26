"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python requirements files.
"""
from enum import StrEnum
from pathlib import Path
from typing import List, Optional, override

from core.build_unit import BuildUnit
from core.modules import Module, ModuleRegistry
from util.files import FileSearch
from util.pip import RequirementsFile, RequirementsTextFile

from targets.dependencies.python.pyproject_toml_file import PyprojectTomlFile
from targets.modules import SubprojectModule


class DependencyType(StrEnum):
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

        @override
        def matches(self, module: Module, _: ModuleRegistry) -> bool:
            return isinstance(module, PythonDependencyModule)

    def __init__(self,
                 dependency_type: DependencyType,
                 root_directory: Path,
                 requirements_file_search: FileSearch = FileSearch()):
        """
        :param dependency_type:             The type of the Python dependencies
        :param root_directory:              The path to the module's root directory
        :param requirements_file_search:    The `FileSearch` that should be used to search for requirements files
        """
        self.dependency_type = dependency_type
        self.root_directory = root_directory
        self.requirements_file_search = requirements_file_search

    def find_requirements_files(self,
                                build_unit: BuildUnit,
                                dependency_type: Optional[DependencyType] = None) -> List[RequirementsFile]:
        """
        Finds and returns all requirements files that belong to the module and optionally match a given
        `DependencyType`.

        :param build_unit:      The `BuildUnit` from which this method is invoked
        :param dependency_type: An optional `DependencyType` to be matched
        :return:                A list that contains the requirements files that have been found
        """
        requirements_files: List[RequirementsFile] = []

        if not dependency_type or self.dependency_type == dependency_type:
            requirements_files.extend([
                RequirementsTextFile(file)
                for file in self.requirements_file_search.filter_by_name('requirements.txt').list(self.root_directory)
            ])

        if dependency_type == DependencyType.BUILD_TIME:
            pyproject_toml_file = self.root_directory / 'pyproject.template.toml'

            if pyproject_toml_file.is_file():
                requirements_files.append(PyprojectTomlFile(build_unit, pyproject_toml_file))

        return requirements_files

    @override
    @property
    def subproject_name(self) -> str:
        return self.root_directory.name

    @override
    def __str__(self) -> str:
        return ('PythonDependencyModule {dependency_type="' + self.dependency_type + '", root_directory="'
                + str(self.root_directory) + '"}')
