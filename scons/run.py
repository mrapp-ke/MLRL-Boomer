"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for installing and running external programs during the build process.
"""
import subprocess
import sys

from functools import reduce
from typing import List, Optional, Tuple

from modules import BUILD_MODULE
from pkg_resources import DistributionNotFound, VersionConflict, parse_requirements, require


def __run_command(cmd: str, *args, print_args: bool = False):
    cmd_formatted = cmd + (reduce(lambda aggr, argument: aggr + ' ' + argument, args, '') if print_args else '')
    print('Running external command "' + cmd_formatted + '"...')
    cmd_args = [cmd]

    for arg in args:
        cmd_args.append(str(arg))

    out = subprocess.run(cmd_args, check=False)
    exit_code = out.returncode

    if exit_code != 0:
        print('External command "' + cmd_formatted + '" terminated with non-zero exit code ' + str(exit_code))
        sys.exit(exit_code)


def __is_dependency_missing(dependency: str) -> bool:
    try:
        require(dependency)
        return False
    except DistributionNotFound:
        return True
    except VersionConflict:
        return False


def __is_dependency_outdated(dependency: str) -> bool:
    try:
        require(dependency)
        return False
    except DistributionNotFound:
        return False
    except VersionConflict:
        return True


def __find_dependencies(requirements_file: str, *dependencies: str) -> List[str]:
    with open(requirements_file, mode='r', encoding='utf-8') as file:
        dependency_dict = {dependency.key: str(dependency) for dependency in parse_requirements(file.read())}

    if dependencies:
        return [dependency_dict[dependency] for dependency in dependencies if dependency in dependency_dict]

    return list(dependency_dict.values())


def __find_missing_and_outdated_dependencies(requirements_file: str, *dependencies: str) -> Tuple[List[str], List[str]]:
    dependencies = __find_dependencies(requirements_file, *dependencies)
    missing_dependencies = [dependency for dependency in dependencies if __is_dependency_missing(dependency)]
    outdated_dependencies = [dependency for dependency in dependencies if __is_dependency_outdated(dependency)]
    return missing_dependencies, outdated_dependencies


def __pip_install(dependencies: List[str], force_reinstall: bool = False):
    args = ['--prefer-binary']

    if force_reinstall:
        args.append('--force-reinstall')

    __run_command('python', '-m', 'pip', 'install', *args, *dependencies, print_args=True)


def __install_dependencies(requirements_file: str, *dependencies: str):
    missing_dependencies, outdated_dependencies = __find_missing_and_outdated_dependencies(
        requirements_file, *dependencies)

    if missing_dependencies:
        __pip_install(missing_dependencies)

    if outdated_dependencies:
        __pip_install(outdated_dependencies, force_reinstall=True)


def install_build_dependencies(*dependencies: str):
    """
    Installs one or several dependencies that are required by the build system.

    :param dependencies: The names of the dependencies that should be installed
    """
    __install_dependencies(BUILD_MODULE.requirements_file, *dependencies)


def run_program(program: str,
                *args,
                print_args: bool = False,
                additional_dependencies: Optional[List[str]] = None,
                requirements_file: str = BUILD_MODULE.requirements_file):
    """
    Runs an external program.

    :param program:                 The name of the program to be run
    :param args:                    Optional arguments that should be passed to the program
    :param print_args:              True, if the arguments should be included in log statements, False otherwise
    :param additional_dependencies: The names of dependencies that should be installed before running the program
    :param requirements_file:       The path of the requirements.txt file that specifies the dependency versions
    """
    dependencies = [program]

    if additional_dependencies:
        dependencies.extend(additional_dependencies)

    __install_dependencies(requirements_file, *dependencies)
    __run_command(program, *args, print_args=print_args)
