"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for installing Python packages via pip.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Optional, Set

from core.build_unit import BuildUnit
from util.cmd import Command as Cmd
from util.io import TextFile
from util.log import Log


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

    def __str__(self) -> str:
        return self.normalized_name

    def __eq__(self, other: 'Package') -> bool:
        return self.normalized_name == other.normalized_name

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

    def __str__(self) -> str:
        return str(self.package) + (' ' + str(self.version) if self.version else '')

    def __eq__(self, other: 'Requirement') -> bool:
        return self.package == other.package

    def __hash__(self) -> int:
        return hash(self.package)


class Requirements(ABC):
    """
    An abstract base class for all classes that provide access to requirements.
    """

    @property
    @abstractmethod
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        """
        A dictionary that contains all requirements by their package.
        """

    @property
    def requirements(self) -> Set[Requirement]:
        """
        A set that contains all requirements in the requirements file
        """
        return set(self.requirements_by_package.values())

    def lookup_requirements(self, *packages: Package, accept_missing: bool = False) -> Set[Requirement]:
        """
        Looks up the requirements for given packages in the requirements file.

        :param packages:        The packages that should be looked up
        :param accept_missing:  False, if an error should be raised if a package is not listed in the requirements file,
                                True, if it should simply be ignored
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
        Looks up the requirement for a given package in the requirements file.

        :param package:         The package that should be looked up
        :param accept_missing:  False, if an error should be raised if the package is not listed in the requirements
                                file, True, if it should simply be ignored
        :return:                The requirement for the given package
        """
        requirements = self.lookup_requirements(package, accept_missing=accept_missing)
        return requirements.pop() if requirements else None


class RequirementsFile(TextFile, Requirements):
    """
    Represents a specific requirements.txt file.
    """

    @property
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        return {
            requirement.package: requirement
            for requirement in
            [Requirement.parse(line.strip('\n').strip()) for line in self.lines if line.strip('\n').strip()]
        }

    def update(self, updated_requirement: Requirement):
        """
        Updates a given requirement, if it is included in the requirements file.

        :param updated_requirement: The requirement to be updated
        """
        new_lines = []

        for line in self.lines:
            new_lines.append(line)
            line_stripped = line.strip('\n').strip()

            if line_stripped:
                requirement = Requirement.parse(line_stripped)

                if requirement.package == updated_requirement.package:
                    new_lines[-1] = str(updated_requirement) + '\n'

        self.write_lines(*new_lines)


class RequirementsFiles(Requirements):
    """
    Represents multiple requirements.txt files.
    """

    def __init__(self, *requirements_files: str):
        self.requirements_files = [RequirementsFile(requirements_file) for requirements_file in requirements_files]

    @property
    def requirements_by_package(self) -> Dict[Package, Requirement]:
        return reduce(lambda aggr, requirements_file: aggr | requirements_file.requirements_by_package,
                      self.requirements_files, {})

    def lookup_requirements_by_file(self,
                                    *packages: Package,
                                    accept_missing: bool = False) -> Dict[str, Set[Requirement]]:
        """
        Looks up the requirements for given packages in the requirements files.

        :param packages:        The packages that should be looked up
        :param accept_missing:  False, if an error should be raised if a package is not listed in any requirements file,
                                True, if it should simply be ignored
        :return:                A dictionary that contains the paths to requirements files, as well as their
                                requirements for the given packages
        """
        requirements_by_file = {}

        for package in packages:
            found = False

            for requirements_file in self.requirements_files:
                requirement = requirements_file.requirements_by_package.get(package)

                if requirement:
                    requirements = requirements_by_file.setdefault(requirements_file.file, set())
                    requirements.add(requirement)
                    found = True

            if not found and not accept_missing:
                raise RuntimeError('Requirement for package "' + str(package) + '" not found')

        return requirements_by_file

    def lookup_requirement_by_file(self, package: Package, accept_missing: bool = False) -> Dict[str, Requirement]:
        """
        Looks up the requirement for a given package in the requirements files.

        :param package:         The package that should be looked up
        :param accept_missing:  False, if an error should be raised if the package is not listed any requirements file,
                                True, if it should simply be ignored
        :return:                A dictionary that contains the paths to requirements files, as well as their requirement
                                for the given package
        """
        requirements_by_file = self.lookup_requirements_by_file(package, accept_missing=accept_missing)
        return {
            requirements_file: requirements.pop()
            for requirements_file, requirements in requirements_by_file.items() if requirements
        }

    def update(self, updated_requirement: Requirement):
        """
        Updates a given requirement, if it is included in one of the requirements files.

        :param updated_requirement: The requirement to be updated
        """
        for requirements_file in self.requirements_files:
            requirements_file.update(updated_requirement)


class Pip:
    """
    Allows to install Python packages via pip.
    """

    class Command(Cmd, ABC):
        """
        An abstract base class for all classes that allow to run pip on the command line.
        """

        def __init__(self, pip_command: str, *arguments: str):
            """
            :param pip_command: The pip command to be run, e.g., "install"
            :param arguments:   Optional arguments to be passed to pip
            """
            super().__init__('python', '-m', 'pip', pip_command, *arguments, '--disable-pip-version-check')

    class InstallCommand(Command):
        """
        Allows to install requirements via the command `pip install`.
        """

        def __init__(self, *requirements: Requirement, dry_run: bool = False):
            """
            :param requirement: The requirements to be installed
            :param dry_run:     True, if the --dry-run flag should be set, False otherwise
            """
            super().__init__('install', *[str(requirement) for requirement in requirements], '--upgrade',
                             '--upgrade-strategy', 'eager', '--prefer-binary')
            self.add_conditional_arguments(dry_run, '--dry-run')

    @staticmethod
    def __would_install_requirements(stdout: str, *requirements: Requirement) -> bool:
        prefix = 'Would install'

        for line in stdout.split('\n'):
            if line.strip().startswith(prefix):
                package = Package(line[len(prefix):].strip())

                for requirement in requirements:
                    if package.normalized_name.find(requirement.package.normalized_name) >= 0:
                        return True

        return False

    @staticmethod
    def install_requirements(*requirements: Requirement, dry_run: bool = False):
        """
        Installs one or several requirements.

        :param requirements:    The requirements to be installed
        :param dry_run:         True, if pip's "--dry-run" option should be set, False otherwise
        """
        if requirements:
            try:
                stdout = Pip.InstallCommand(*requirements, dry_run=dry_run) \
                    .print_command(False) \
                    .exit_on_error(not dry_run) \
                    .capture_output()

                if Pip.__would_install_requirements(stdout, *requirements):
                    if dry_run:
                        Pip.InstallCommand(*requirements) \
                            .print_arguments(True) \
                            .run()
                    else:
                        Log.info(stdout)
            except RuntimeError:
                Pip.install_requirements(*requirements)

    def __init__(self, *requirements_files: str):
        """
        :param requirements_files: The paths to the requirements files that specify the versions of the packages to be
                                   installed
        """
        self.requirements = RequirementsFiles(*requirements_files)

    @staticmethod
    def for_build_unit(build_unit: BuildUnit = BuildUnit.for_file(__file__)):
        """
        Creates and returns a new `Pip` instance for installing packages for a specific build unit.

        :param build_unit:  The build unit for which packages should be installed
        :return:            The `Pip` instance that has been created
        """
        return Pip(*build_unit.find_requirements_files())

    def install_packages(self, *package_names: str, accept_missing: bool = False):
        """
        Installs one or several dependencies in the requirements file.

        :param package_names:   The names of the packages that should be installed
        :param accept_missing:  False, if an error should be raised if a package is not listed in the requirements file,
                                True, if it should simply be ignored
        """
        packages = [Package(package_name) for package_name in package_names]
        requirements = self.requirements.lookup_requirements(*packages, accept_missing=accept_missing)
        self.install_requirements(*requirements, dry_run=True)
