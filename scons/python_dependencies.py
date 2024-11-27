"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with dependencies.
"""

from dataclasses import dataclass
from os import path
from typing import List

from modules_old import ALL_MODULES, CPP_MODULE, PYTHON_MODULE, Module
from util.pip import Package, Pip, RequirementsFile
from util.units import BuildUnit


@dataclass
class Dependency:
    """
    Provides information about an installed dependency.

    Attributes:
        package:            The Python package
        installed_version:  The version of the dependency that is currently installed
        latest_version:     The latest version of the dependency
    """
    package: Package
    installed_version: str
    latest_version: str


def __find_outdated_dependencies() -> List[Dependency]:
    stdout = Pip.Command('list', '--outdated').print_command(False).capture_output()
    stdout_lines = stdout.strip().split('\n')
    i = 0

    for line in stdout_lines:
        i += 1

        if line.startswith('----'):
            break

    outdated_dependencies = []

    for line in stdout_lines[i:]:
        parts = line.split()
        outdated_dependencies.append(Dependency(Package(parts[0]), installed_version=parts[1], latest_version=parts[2]))

    return outdated_dependencies


def __install_module_dependencies(module: Module, *dependencies: str):
    requirements_file = module.requirements_file

    if path.isfile(requirements_file):
        Pip.for_build_unit(module).install_packages(*dependencies)


def __print_table(header: List[str], rows: List[List[str]]):
    Pip.for_build_unit(BuildUnit('util')).install_packages('tabulate')
    # pylint: disable=import-outside-toplevel
    from tabulate import tabulate
    print(tabulate(rows, headers=header))


def install_runtime_dependencies(**_):
    """
    Installs all runtime dependencies that are required by the Python and C++ module.
    """
    __install_module_dependencies(PYTHON_MODULE)
    __install_module_dependencies(CPP_MODULE)


def check_dependency_versions(**_):
    """
    Installs all dependencies used by the project and checks for outdated dependencies.
    """
    print('Installing all dependencies...')
    for module in ALL_MODULES:
        __install_module_dependencies(module)

    print('Checking for outdated dependencies...')
    outdated_dependencies = __find_outdated_dependencies()
    rows = []

    for dependency in outdated_dependencies:
        for module in ALL_MODULES:
            requirements_file = module.requirements_file

            if path.isfile(requirements_file):
                requirements = RequirementsFile(requirements_file).lookup(dependency.package, accept_missing=True)

                if requirements and requirements.pop().version:
                    rows.append([str(dependency.package), dependency.installed_version, dependency.latest_version])
                    break

    if rows:
        rows.sort(key=lambda row: row[0])
        header = ['Dependency', 'Installed version', 'Latest version']
        print('The following dependencies are outdated:\n')
        __print_table(header=header, rows=rows)
    else:
        print('All dependencies are up-to-date!')
