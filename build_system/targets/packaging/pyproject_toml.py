"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing pyproject.toml files.
"""
from functools import reduce
from typing import List

from util.toml_file import TomlFile


class PyprojectTomlFile(TomlFile):
    """
    A pyproject.toml file.
    """

    @property
    def package_name(self) -> str:
        """
        The package name declared in the pyproject.toml file.
        """
        return self.toml_dict['project']['name']

    @property
    def mandatory_dependencies(self) -> List[str]:
        """
        A list that contains the mandatory dependencies declared in the pyproject.toml file.
        """
        return self.toml_dict['project'].get('dependencies', [])

    @property
    def optional_dependencies(self) -> List[str]:
        """
        A list that contains the optional dependencies declared in the pyproject.toml file.
        """
        dependency_dict = self.toml_dict['project'].get('optional-dependencies', {})
        return reduce(lambda aggr, dependency_list: aggr + dependency_list, dependency_dict.values(), [])

    @property
    def dependencies(self) -> List[str]:
        """
        A list that contains all dependencies, including mandatory and optional ones, declared in the pyproject.toml
        file.
        """
        return self.mandatory_dependencies + self.optional_dependencies
