"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for installing Python packages via pip.
"""
from abc import ABC
from functools import reduce
from typing import Set

from util.cmd import Command as Cmd
from util.requirements import Package, Requirement, RequirementsFiles


class PackageManager:
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
    def install_requirements(*requirements: Requirement, silent: bool = False):
        """
        Installs one or several requirements.

        :param requirements:    The requirements to be installed
        :param silent:          True, if any log output should be suppressed, False otherwise
        """
        if requirements:
            stdout = PackageManager.InstallCommand(*requirements, dry_run=True) \
                .print_command(False) \
                .exit_on_error(False) \
                .capture_output()

            if PackageManager.__would_install_requirements(stdout):
                install_command = PackageManager.InstallCommand(*requirements)

                if silent:
                    install_command.capture_output()
                else:
                    install_command.print_arguments(True).run()

    @staticmethod
    def install_packages(requirements_files: RequirementsFiles,
                         *package_names: str,
                         accept_missing: bool = False,
                         silent: bool = False):
        """
        Installs one or several dependencies.

        :param requirements_files: The requirements files that specify the versions of the packages to be installed
        :param package_names:       The names of the packages that should be installed
        :param accept_missing:      False, if an error should be raised if the requirement for a package is not found,
                                    True, if it should simply be ignored
        :param silent:              True, if any log output should be suppressed, False otherwise
        """
        looked_up_requirements = requirements_files.lookup_requirements(*package_names, accept_missing=accept_missing)
        requirements_to_be_installed: Set[Requirement] = set()
        requirements_to_be_installed = reduce(lambda aggr, requirements: aggr | requirements,
                                              looked_up_requirements.values(), requirements_to_be_installed)
        PackageManager.install_requirements(*requirements_to_be_installed, silent=silent)

    @staticmethod
    def install_all_packages(requirements_files: RequirementsFiles):
        """
        Installs all dependencies in the requirements file.
        """
        requirements: Set[Requirement] = set()
        requirements = reduce(lambda aggr, requirements_file: aggr | requirements_file.requirements, requirements_files,
                              requirements)
        PackageManager.install_requirements(*requirements)
