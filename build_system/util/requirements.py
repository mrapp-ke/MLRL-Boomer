"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for dealing with Python dependencies via requirements.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, override

from core.build_unit import BuildUnit
from util.format import format_iterable
from util.io import TextFile


@dataclass
class Package:
    """
    A Python package.

    Attributes:
        name: The name of the package
    """
    name: str

    @property
    def normalized_name(self) -> str:
        """
        The normalized name of the package in lower-case and with invalid characters being replaced.
        """
        return self.name.replace('_', '-').lower()

    @override
    def __str__(self) -> str:
        return self.normalized_name

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.normalized_name == other.normalized_name

    @override
    def __hash__(self) -> int:
        return hash(self.normalized_name)


@dataclass
class RequirementVersion:
    """
    Specifies the version of a requirement.

    Attributes:
        min_version:    The maximum version number
        max_version:    The minimum version number
    """
    min_version: str
    max_version: str

    PREFIX_EQ = '=='

    PREFIX_GEQ = '>='

    PREFIX_LE = '<'

    @staticmethod
    def parse(version: str) -> 'RequirementVersion':
        """
        Parses and returns the version of a requirement.

        :param version: The version to be parsed
        :return:        The version that has been parsed
        """
        parts = version.strip().split(',')

        if len(parts) > 2:
            raise ValueError(
                'Version of requirement must consist of one or two version numbers, separated by comma, but got: '
                + version)

        first_part = parts[0].strip()

        if len(parts) > 1:
            if not first_part.startswith(RequirementVersion.PREFIX_GEQ):
                raise ValueError('First version number of requirement must start with "' + RequirementVersion.PREFIX_GEQ
                                 + '", but got: ' + version)

            second_part = parts[1].strip()

            if not second_part.startswith(RequirementVersion.PREFIX_LE):
                raise ValueError('Second version number of requirement must start with "' + RequirementVersion.PREFIX_LE
                                 + '", but got: ' + version)

            return RequirementVersion(min_version=first_part[len(RequirementVersion.PREFIX_GEQ):].strip(),
                                      max_version=second_part[len(RequirementVersion.PREFIX_LE):].strip())

        version_number = first_part

        if version_number.startswith(RequirementVersion.PREFIX_EQ):
            version_number = version_number[len(RequirementVersion.PREFIX_EQ):].strip()

        return RequirementVersion(min_version=version_number, max_version=version_number)

    def is_range(self) -> bool:
        """
        Returns whether the version specifies a range of version numbers or not.

        :return: True, if the version specifies a range of version numbers, False otherwise
        """
        return self.min_version != self.max_version

    @override
    def __str__(self) -> str:
        if self.is_range():
            return self.PREFIX_GEQ + ' ' + self.min_version + ', ' + self.PREFIX_LE + ' ' + self.max_version
        return self.PREFIX_EQ + ' ' + self.min_version


@dataclass
class Requirement:
    """
    A single requirement included in a requirements file, consisting of a Python package and an optional version.

    Attributes:
        package:    The package
        version:    The version of the package or None, if no version is specified
    """
    package: Package
    version: Optional[RequirementVersion] = None

    @staticmethod
    def parse(requirement: str) -> 'Requirement':
        """
        Parses and returns a single requirement included a requirements file.

        :param requirement: The requirement to be parsed
        :return:            The requirement that has been parsed
        """
        parts = requirement.split()
        package = Package(name=parts[0].strip())
        version = RequirementVersion.parse(' '.join(parts[1:])) if len(parts) > 1 else None
        return Requirement(package, version)

    @override
    def __str__(self) -> str:
        return str(self.package) + (' ' + str(self.version) if self.version else '')

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.package == other.package

    @override
    def __hash__(self) -> int:
        return hash(self.package)


class RequirementsFile(ABC):
    """
    An abstract base class for all classes that provide access to requirements stored in a file.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """
        The path to the file.
        """

    @property
    @abstractmethod
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        """
        A dictionary that contains all requirements in the file by their package.
        """

    @property
    def requirements(self) -> Set[Requirement]:
        """
        A set that contains all requirements in the file.
        """
        return set(self.requirements_by_package.values())

    @abstractmethod
    def update(self, outdated_requirement: Requirement, updated_requirement: Requirement):
        """
        Updates a given requirement, if it is included in the requirements file.

        :param outdated_requirement:    The outdated requirement
        :param updated_requirement:     The requirement to be updated
        """

    def lookup_requirements(self, *packages: Package, accept_missing: bool = False) -> Set[Requirement]:
        """
        Looks up the requirements for given packages.

        :param packages:        The packages that should be looked up
        :param accept_missing:  False, if an error should be raised if the requirement for a package is not found, True,
                                if it should simply be ignored
        :return:                A set that contains the requirements for the given packages
        """
        requirements = set()

        for package in packages:
            requirement = self.requirements_by_package.get(package)

            if requirement:
                requirements.add(requirement)
            elif not accept_missing:
                raise RuntimeError('Requirement for package "' + str(package) + '" not found')

        return requirements

    def lookup_requirement(self, package: Package, accept_missing: bool = False) -> Optional[Requirement]:
        """
        Looks up the requirement for a given package.

        :param package:         The package that should be looked up
        :param accept_missing:  False, if an error should be raised if the requirement for the package is not found,
                                True, if it should simply be ignored
        :return:                The requirement for the given package
        """
        requirements = self.lookup_requirements(package, accept_missing=accept_missing)
        return requirements.pop() if requirements else None

    @override
    def __str__(self) -> str:
        return str(self.path)

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    @override
    def __hash__(self) -> int:
        return hash(self.path)


class RequirementsTextFile(TextFile, RequirementsFile):
    """
    Represents a specific requirements.txt file.
    """

    @override
    @property
    def path(self) -> Path:
        return self.file

    @override
    @property
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        return {
            requirement.package: requirement
            for requirement in
            [Requirement.parse(line.strip('\n').strip()) for line in self.lines if line.strip('\n').strip()]
        }

    @override
    def update(self, _: Requirement, updated_requirement: Requirement):
        new_lines = []

        for line in self.lines:
            new_lines.append(line)
            line_stripped = line.strip('\n').strip()

            if line_stripped:
                requirement = Requirement.parse(line_stripped)

                if requirement.package == updated_requirement.package:
                    new_lines[-1] = str(updated_requirement) + '\n'

        self.write_lines(*new_lines)


class RequirementsFiles(Iterable[RequirementsFile]):
    """
    Represents multiple requirements files that specify the versions of the packages to be installed.
    """

    def __init__(self, *requirements_files: RequirementsFile):
        """
        :param requirements_files: The requirements files that specify the versions of the packages to be installed
        """
        self._requirements_files = list(requirements_files)

    @staticmethod
    def for_build_unit(build_unit: BuildUnit = BuildUnit.for_file(Path(__file__))):
        """
        Creates and returns a new `RequirementsFiles` instance for installing packages for a specific build unit.

        :param build_unit:  The build unit for which packages should be installed
        :return:            The `RequirementsFiles` instance that has been created
        """
        return RequirementsFiles(*[RequirementsTextFile(file) for file in build_unit.find_requirements_files()])

    def lookup_requirements(self,
                            *package_names: str,
                            accept_missing: bool = False) -> Dict[RequirementsFile, Set[Requirement]]:
        """
        Looks up the requirements for given packages.

        :param package_names:   The names of the packages that should be looked up
        :param accept_missing:  False, if an error should be raised if the requirement for a package is not found, True,
                                if it should simply be ignored
        :return:                A dictionary that contains requirement files, as well as their requirements for the
                                given packages
        """
        packages = [Package(package_name) for package_name in package_names]
        missing_package_names = {package.normalized_name for package in packages}
        result: Dict[RequirementsFile, Set[Requirement]] = {}

        for requirements_file in self._requirements_files:
            requirements = requirements_file.lookup_requirements(*packages, accept_missing=True)

            for requirement in requirements:
                missing_package_names.discard(requirement.package.normalized_name)
                result.setdefault(requirements_file, set()).add(requirement)

        if missing_package_names and not accept_missing:
            raise RuntimeError('Requirements for packages ' + format_iterable(missing_package_names, delimiter='"')
                               + ' not found')

        return result

    def lookup_requirement(self,
                           package_name: str,
                           accept_missing: bool = False) -> Dict[RequirementsFile, Requirement]:
        """
        Looks up the requirement for a given package.

        :param package_name:    The name of the package that should be looked up
        :param accept_missing:  False, if an error should be raised if the requirement for the package is not found,
                                True, if it should simply be ignored
        :return:                A dictionary that contains requirement files, as well as their requirements for the
                                given packages
        """
        looked_up_requirements = self.lookup_requirements(package_name, accept_missing=accept_missing)
        return {
            requirements_file: requirements.pop()
            for requirements_file, requirements in looked_up_requirements.items() if requirements
        }

    @override
    def __iter__(self) -> Iterator[RequirementsFile]:
        return iter(self._requirements_files)
