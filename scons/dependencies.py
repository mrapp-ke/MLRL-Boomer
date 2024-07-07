"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with dependencies.
"""

from dataclasses import dataclass
from functools import reduce
from os import path
from typing import List, Optional

from command_line import run_command
from modules import ALL_MODULES, BUILD_MODULE, CPP_MODULE, PYTHON_MODULE, Module


@dataclass
class Requirement:
    """
    Specifies the supported version(s) of a specific dependency.

    Attributes:
        dependency: The name of the dependency
        version:    The supported version(s) of the dependency or None, if there are no restrictions
    """
    dependency: str
    version: Optional[str] = None

    def __str__(self):
        return self.dependency + (self.version if self.version else '')


def __run_pip_command(*args, **kwargs):
    return run_command('python', '-m', 'pip', *args, **kwargs)


def __run_pip_install_command(requirement: Requirement, *args, **kwargs):
    return __run_pip_command('install', str(requirement), '--upgrade', '--upgrade-strategy', 'eager', '--prefer-binary',
                             '--disable-pip-version-check', *args, **kwargs)


def __pip_install(requirement: Requirement, dry_run: bool = False):
    try:
        args = ['--dry-run'] if dry_run else []
        out = __run_pip_install_command(requirement,
                                        *args,
                                        print_cmd=False,
                                        capture_output=True,
                                        exit_on_error=not dry_run)
        stdout = str(out.stdout).strip()
        stdout_lines = stdout.split('\n')

        if reduce(
                lambda aggr, line: aggr | line.startswith('Would install') and __normalize_dependency(line).find(
                    requirement.dependency) >= 0, stdout_lines, False):
            if dry_run:
                __run_pip_install_command(requirement, print_args=True)
            else:
                print(stdout)
    except RuntimeError:
        __pip_install(requirement)


def __normalize_dependency(dependency: str):
    return dependency.replace('_', '-').lower()


def __find_requirements(requirements_file: str, *dependencies: str, raise_error: bool = True) -> List[Requirement]:
    with open(requirements_file, mode='r', encoding='utf-8') as file:
        lines = [line.split(' ') for line in file.readlines()]
        requirements = [
            Requirement(dependency=__normalize_dependency(parts[0].strip()),
                        version=' '.join(parts[1:]).strip() if len(parts) > 1 else None) for parts in lines
        ]
        requirements = {requirement.dependency: requirement for requirement in requirements}

    if dependencies:
        found_requirements = []

        for dependency in dependencies:
            if __normalize_dependency(dependency) in requirements:
                found_requirements.append(requirements[dependency])
            elif raise_error:
                raise RuntimeError('Dependency "' + dependency + '" not found in requirements file "'
                                   + requirements_file + '"')

        return found_requirements

    return list(requirements.values())


def __install_module_dependencies(module: Module, *dependencies: str):
    requirements_file = module.requirements_file

    if path.isfile(requirements_file):
        install_dependencies(requirements_file, *dependencies)


def install_dependencies(requirements_file: str, *dependencies: str):
    """
    Installs one or several dependencies if they are listed in a given requirements.txt file.

    :param requirements_file:   The path of the requirements.txt file that specifies the dependency versions
    :param dependencies:        The names of the dependencies that should be installed
    """
    for requirement in __find_requirements(requirements_file, *dependencies):
        __pip_install(requirement, dry_run=True)


def install_build_dependencies(*dependencies: str):
    """
    Installs one or several dependencies that are required by the build system.

    :param dependencies: The names of the dependencies that should be installed
    """
    __install_module_dependencies(BUILD_MODULE, *dependencies)


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
    out = __run_pip_command('list', '--outdated', print_cmd=False, capture_output=True)
    stdout = str(out.stdout).strip()
    stdout_lines = stdout.split('\n')
    i = 0

    for line in stdout_lines:
        i += 1

        if line.startswith('----'):
            break

    outdated_dependencies = []

    for line in stdout_lines[i:]:
        dependency = line.split()[0]

        for module in ALL_MODULES:
            requirements_file = module.requirements_file

            if path.isfile(requirements_file):
                requirements = __find_requirements(requirements_file, dependency, raise_error=False)

                if requirements and requirements[0].version:
                    outdated_dependencies.append(line)
                    break

    if outdated_dependencies:
        print('The following dependencies are outdated:\n')

        for header_line in stdout_lines[:i]:
            print(header_line)

        for outdated_dependency in outdated_dependencies:
            print(outdated_dependency)
    else:
        print('All dependencies are up-to-date!')
