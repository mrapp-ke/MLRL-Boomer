"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing installed Python dependencies via pip.
"""
from dataclasses import dataclass
from typing import Set

from util.pip import Package, Pip, Requirement, RequirementVersion


@dataclass
class Dependency:
    """
    Provides information about an outdated dependency.

    Attributes:
        requirements_file:  The path to the requirements file that defines the dependency
        package:            The package, the dependency corresponds to
        outdated:           The outdated version of the dependency
        latest:             The latest version of the dependency
    """
    requirements_file: str
    package: Package
    outdated: RequirementVersion
    latest: RequirementVersion

    def __eq__(self, other: 'Dependency') -> bool:
        return self.requirements_file == other.requirements_file and self.package == other.package

    def __hash__(self) -> int:
        return hash((self.requirements_file, self.package))


class PipList(Pip):
    """
    Allows to list installed Python packages via pip.
    """

    class ListCommand(Pip.Command):
        """
        Allows to list information about installed packages via the command `pip list`.
        """

        def __init__(self, outdated: bool = False):
            """
            :param outdated: True, if only outdated packages should be listed, False otherwise
            """
            super().__init__('list')
            self.add_conditional_arguments(outdated, '--outdated')

    def install_all_packages(self):
        """
        Installs all dependencies in the requirements file.
        """
        for requirement in self.requirements.requirements:
            Pip.install_requirement(requirement, dry_run=True)

    def list_outdated_dependencies(self) -> Set[Dependency]:
        """
        Returns all outdated Python dependencies that are currently installed.

        :return: A set that contains all outdated dependencies
        """
        stdout = PipList.ListCommand(outdated=True).print_command(False).capture_output()
        stdout_lines = stdout.strip().split('\n')
        i = 0

        for line in stdout_lines:
            i += 1

            if line.startswith('----'):
                break

        outdated_dependencies = set()

        for line in stdout_lines[i:]:
            parts = line.split()

            if len(parts) < 3:
                raise ValueError(
                    'Output of command "pip list" is expected to be a table with at least three columns, but got:'
                    + line)

            package = Package(parts[0])
            requirements_by_file = self.requirements.lookup_requirement_by_file(package, accept_missing=True)

            for requirements_file, requirement in requirements_by_file.items():
                if requirement.version:
                    installed_version = parts[1]
                    latest_version = parts[2]
                    outdated_dependencies.add(
                        Dependency(requirements_file=requirements_file,
                                   package=package,
                                   outdated=RequirementVersion.parse(installed_version),
                                   latest=RequirementVersion.parse(latest_version)))

        return outdated_dependencies
