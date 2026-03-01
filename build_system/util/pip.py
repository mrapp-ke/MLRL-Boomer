"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for installing Python packages via pip.
"""
from abc import ABC
from functools import reduce
from pathlib import Path
from typing import Dict, Set

from core.build_unit import BuildUnit
from util.cmd import Command as Cmd
from util.format import format_iterable
from util.requirements import Package, Requirement, RequirementsFile, RequirementsTextFile


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

    def __init__(self, *requirements_files: RequirementsFile):
        """
        :param requirements_files: The requirements files that specify the versions of the packages to be installed
        """
        self.requirements_files = list(requirements_files)

    @staticmethod
    def for_build_unit(build_unit: BuildUnit = BuildUnit.for_file(Path(__file__))):
        """
        Creates and returns a new `Pip` instance for installing packages for a specific build unit.

        :param build_unit:  The build unit for which packages should be installed
        :return:            The `Pip` instance that has been created
        """
        return Pip(*[RequirementsTextFile(file) for file in build_unit.find_requirements_files()])

    @staticmethod
    def install_requirements(*requirements: Requirement, silent: bool = False):
        """
        Installs one or several requirements.

        :param requirements:    The requirements to be installed
        :param silent:          True, if any log output should be suppressed, False otherwise
        """
        if requirements:
            stdout = Pip.InstallCommand(*requirements, dry_run=True) \
                .print_command(False) \
                .exit_on_error(False) \
                .capture_output()

            if Pip.__would_install_requirements(stdout, *requirements):
                install_command = Pip.InstallCommand(*requirements)

                if silent:
                    install_command.capture_output()
                else:
                    install_command.print_arguments(True).run()

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

        for requirements_file in self.requirements_files:
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

    def install_packages(self, *package_names: str, accept_missing: bool = False, silent: bool = False):
        """
        Installs one or several dependencies.

        :param package_names:   The names of the packages that should be installed
        :param accept_missing:  False, if an error should be raised if the requirement for a package is not found, True,
                                if it should simply be ignored
        :param silent:          True, if any log output should be suppressed, False otherwise
        """
        looked_up_requirements = self.lookup_requirements(*package_names, accept_missing=accept_missing)
        requirements_to_be_installed: Set[Requirement] = set()
        requirements_to_be_installed = reduce(lambda aggr, requirements: aggr | requirements,
                                              looked_up_requirements.values(), requirements_to_be_installed)
        self.install_requirements(*requirements_to_be_installed, silent=silent)

    def install_all_packages(self):
        """
        Installs all dependencies in the requirements file.
        """
        requirements: Set[Requirement] = set()
        requirements = reduce(lambda aggr, requirements_file: aggr | requirements_file.requirements,
                              self.requirements_files, requirements)
        Pip.install_requirements(*requirements)
