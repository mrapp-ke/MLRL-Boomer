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

    def update(self, outdated_requirement: Requirement, updated_requirement: Requirement):
        outdated_requirement_string = str(outdated_requirement)
        new_lines = []

        for line in self.lines:
            if line.find(outdated_requirement_string) >= 0:
                line = line.replace(outdated_requirement_string, str(updated_requirement))

            new_lines.append(line)

        self.write_lines(*new_lines)
