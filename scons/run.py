"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for installing and running external programs during the build process.
"""
import subprocess
import sys

from functools import reduce
from os import path
from typing import List, Optional

from modules import BUILD_MODULE, CPP_MODULE, PYTHON_MODULE


def __format_command(cmd: str, *args, format_args: bool = True) -> str:
    return cmd + (reduce(lambda aggr, argument: aggr + ' ' + argument, args, '') if format_args else '')


def __run_command(cmd: str,
                  *args,
                  print_cmd: bool = True,
                  print_args: bool = False,
                  capture_output: bool = False,
                  exit_on_error: bool = True):
    if print_cmd:
        print('Running external command "' + __format_command(cmd, *args, format_args=print_args) + '"...')

    out = subprocess.run([cmd] + list(args), check=False, text=capture_output, capture_output=capture_output)
    exit_code = out.returncode

    if exit_code != 0:
        message = ('External command "' + __format_command(cmd, *args) + '" terminated with non-zero exit code '
                   + str(exit_code))

        if exit_on_error:
            if capture_output:
                print(str(out.stderr).strip())

            print(message)
            sys.exit(exit_code)
        else:
            raise RuntimeError(message)

    return out


def __run_pip_command(requirement: str, *args, **kwargs):
    return __run_command('python', '-m', 'pip', 'install', requirement, '--upgrade', '--prefer-binary',
                         '--disable-pip-version-check', *args, **kwargs)


def __pip_install(requirement: str, dry_run: bool = False):
    try:
        args = ['--dry-run'] if dry_run else []
        out = __run_pip_command(requirement, *args, print_cmd=False, capture_output=True, exit_on_error=not dry_run)
        stdout = str(out.stdout).strip()
        stdout_lines = stdout.split('\n')

        if not reduce(lambda aggr, line: aggr & line.startswith('Requirement already satisfied'), stdout_lines, True):
            if dry_run:
                __run_pip_command(requirement)
            else:
                print(stdout)
    except RuntimeError:
        __pip_install(requirement)


def __find_requirements(requirements_file: str, *dependencies: str) -> List[str]:
    with open(requirements_file, mode='r', encoding='utf-8') as file:
        requirements = {line.split(' ')[0]: line.strip() for line in file.readlines()}

    if dependencies:
        return [requirements[dependency] for dependency in dependencies if dependency in requirements]

    return list(requirements.values())


def __install_dependencies(requirements_file: str, *dependencies: str):
    for requirement in __find_requirements(requirements_file, *dependencies):
        __pip_install(requirement, dry_run=True)


def install_build_dependencies(*dependencies: str):
    """
    Installs one or several dependencies that are required by the build system.

    :param dependencies: The names of the dependencies that should be installed
    """
    __install_dependencies(BUILD_MODULE.requirements_file, *dependencies)


def install_runtime_dependencies(**_):
    """
    Installs all runtime dependencies that are required by the Python and C++ module.
    """
    for module in [PYTHON_MODULE, CPP_MODULE]:
        requirements_file = module.requirements_file

        if path.isfile(requirements_file):
            __install_dependencies(requirements_file)


def run_program(program: str,
                *args,
                print_args: bool = False,
                additional_dependencies: Optional[List[str]] = None,
                requirements_file: str = BUILD_MODULE.requirements_file):
    """
    Runs an external program that has been installed into the virtual environment.

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


def run_python_program(program: str,
                       *args,
                       print_args: bool = False,
                       additional_dependencies: Optional[List[str]] = None,
                       requirements_file: str = BUILD_MODULE.requirements_file):
    """
    Runs an external Python program.

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
    __run_command(path.join(path.dirname(sys.executable), 'python'), '-m', program, *args, print_args=print_args)
