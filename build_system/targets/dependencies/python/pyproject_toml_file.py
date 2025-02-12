"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing pyproject.toml files.
"""
from typing import Dict

from util.pip import Package, Requirement, RequirementsFile
from util.toml_file import TomlFile


class PyprojectTomlFile(TomlFile, RequirementsFile):
    """
    Represents a pyproject.toml file.
    """

    @property
    def path(self) -> str:
        return self.file

    @property
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        requirements = self.toml_dict.get('build-system', {}).get('requires', [])
        return {
            requirement.package: requirement
            for requirement in [Requirement.parse(requirement.strip('\n').strip()) for requirement in requirements]
        }
