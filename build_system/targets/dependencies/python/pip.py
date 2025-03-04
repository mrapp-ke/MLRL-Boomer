"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing installed Python dependencies via pip.
"""
from dataclasses import dataclass, replace
from functools import reduce
from typing import Set

from util.pip import Package, Pip, RequirementsFile, RequirementVersion


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
    requirements_file: RequirementsFile
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
        requirements = reduce(lambda aggr, requirements_file: aggr | requirements_file.requirements,
                              self.requirements_files, set())
        Pip.install_requirements(*requirements)

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

            package_name = parts[0]
            looked_up_requirements = self.lookup_requirement(package_name, accept_missing=True)

            for requirements_file, requirement in looked_up_requirements.items():
                outdated_version = requirement.version

                if outdated_version:
                    latest_version = parts[2]
                    outdated_dependencies.add(
                        Dependency(requirements_file=requirements_file,
                                   package=Package(package_name),
                                   outdated=outdated_version,
                                   latest=RequirementVersion.parse(latest_version)))

        return outdated_dependencies

    def update_outdated_dependencies(self) -> Set[Dependency]:
        """
        Updates all outdated Python dependencies that are currently installed.

        :return: A set that contains all dependencies that have been updated
        """
        updated_dependencies = set()
        separator = '.'

        for outdated_dependency in self.list_outdated_dependencies():
            latest_version = outdated_dependency.latest
            latest_version_parts = [int(part) for part in latest_version.min_version.split(separator)]
            outdated_version = outdated_dependency.outdated
            updated_version = latest_version

            if outdated_version.is_range():
                min_version_parts = [int(part) for part in outdated_version.min_version.split(separator)]
                max_version_parts = [int(part) for part in outdated_version.max_version.split(separator)]
                num_version_numbers = min(len(min_version_parts), len(max_version_parts), len(latest_version_parts))

                for i in range(num_version_numbers):
                    min_version_parts[i] = latest_version_parts[i]
                    max_version_parts[i] = latest_version_parts[i]

                max_version_parts[num_version_numbers - 1] += 1
                updated_version = RequirementVersion(
                    min_version=separator.join([str(part) for part in min_version_parts[:num_version_numbers]]),
                    max_version=separator.join([str(part) for part in max_version_parts[:num_version_numbers]]))

            looked_up_requirements = self.lookup_requirement(outdated_dependency.package.name)

            for requirements_file, outdated_requirement in looked_up_requirements.items():
                requirements_file.update(outdated_requirement, replace(outdated_requirement, version=updated_version))
                updated_dependencies.add(
                    Dependency(requirements_file=requirements_file,
                               package=outdated_dependency.package,
                               outdated=outdated_version,
                               latest=updated_version))

        return updated_dependencies
