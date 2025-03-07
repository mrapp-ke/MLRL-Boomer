"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing installed Python dependencies via pip.
"""
from dataclasses import dataclass, replace
from typing import Set

from core.build_unit import BuildUnit
from util.log import Log
from util.pip import Package, Pip, RequirementsFile, RequirementVersion
from util.version import Version


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

    @staticmethod
    def __query_latest_package_version(build_unit: BuildUnit, package: Package) -> Version:
        Pip.for_build_unit(build_unit).install_packages('requests')
        # pylint: disable=import-outside-toplevel
        import requests
        url = 'https://pypi.org/pypi/' + package.name + '/json'
        Log.info('Querying latest version of package "' + str(package) + '" from ' + url)
        response = requests.get(url, timeout=5)
        latest_version = Version.parse(response.json()['info']['version'], skip_on_error=True)
        Log.info('Latest version of package "' + str(package) + '" is ' + str(latest_version))
        return latest_version

    def list_outdated_dependencies(self, build_unit: BuildUnit) -> Set[Dependency]:
        """
        Returns all outdated Python dependencies that are currently installed.

        :param build_unit:  The `BuildUnit` from which this function is invoked
        :return:            A set that contains all outdated dependencies
        """
        outdated_dependencies = set()
        version_cache = {}

        for requirements_file in self.requirements_files:
            for requirement in requirements_file.requirements:
                package = requirement.package
                current_version = requirement.version
                latest_version = version_cache.get(package)

                if not latest_version:
                    latest_version = self.__query_latest_package_version(build_unit, package)
                    version_cache[package] = latest_version

                if Version.parse(current_version.min_version, skip_on_error=True) < latest_version:
                    outdated_dependencies.add(
                        Dependency(requirements_file=requirements_file,
                                   package=package,
                                   outdated=current_version,
                                   latest=RequirementVersion(min_version=latest_version, max_version=latest_version)))

        return outdated_dependencies

    def update_outdated_dependencies(self, build_unit: BuildUnit) -> Set[Dependency]:
        """
        Updates all outdated Python dependencies that are currently installed.

        :param build_unit:  The `BuildUnit` from which this function is invoked
        :return:            A set that contains all dependencies that have been updated
        """
        updated_dependencies = set()

        for outdated_dependency in self.list_outdated_dependencies(build_unit):
            latest_version = outdated_dependency.latest
            latest_version_numbers = latest_version.min_version.numbers
            outdated_version = outdated_dependency.outdated
            updated_version = latest_version

            if outdated_version.is_range():
                min_version_numbers = list(Version.parse(outdated_version.min_version, skip_on_error=True).numbers)
                max_version_numbers = list(Version.parse(outdated_version.max_version, skip_on_error=True).numbers)
                num_version_numbers = min(len(min_version_numbers), len(max_version_numbers),
                                          len(latest_version_numbers))

                for i in range(num_version_numbers):
                    min_version_numbers[i] = latest_version_numbers[i]
                    max_version_numbers[i] = latest_version_numbers[i]

                max_version_numbers[num_version_numbers - 1] += 1
                updated_version = RequirementVersion(
                    min_version=str(Version(tuple(min_version_numbers[:num_version_numbers]))),
                    max_version=str(Version(tuple(max_version_numbers[:num_version_numbers]))))

            looked_up_requirements = self.lookup_requirement(outdated_dependency.package.name)

            for requirements_file, outdated_requirement in looked_up_requirements.items():
                requirements_file.update(outdated_requirement, replace(outdated_requirement, version=updated_version))
                updated_dependencies.add(
                    Dependency(requirements_file=requirements_file,
                               package=outdated_dependency.package,
                               outdated=outdated_version,
                               latest=updated_version))

        return updated_dependencies
