"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for installing and running external programs during the build process.
"""
import subprocess
import sys

from functools import reduce
from os import path
from typing import List, Optional

from modules import ALL_MODULES, BUILD_MODULE, CPP_MODULE, PYTHON_MODULE, Module


def __format_command(cmd: str, *args, format_args: bool = True) -> str:
    return cmd + (reduce(lambda aggr, argument: aggr + ' ' + argument, args, '') if format_args else '')


def __run_command(cmd: str,
                  *args,
                  print_cmd: bool = True,
                  print_args: bool = False,
                  capture_output: bool = False,
                  exit_on_error: bool = True,
                  env=None):
    if print_cmd:
        print('Running external command "' + __format_command(cmd, *args, format_args=print_args) + '"...')

    out = subprocess.run([cmd] + list(args), check=False, text=capture_output, capture_output=capture_output, env=env)
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


def __run_pip_command(*args, **kwargs):
    return __run_command('python', '-m', 'pip', *args, **kwargs)


def __run_pip_install_command(requirement: str, *args, **kwargs):
    return __run_pip_command('install', requirement, '--upgrade', '--prefer-binary', '--disable-pip-version-check',
                             *args, **kwargs)


def __pip_install(requirement: str, dry_run: bool = False):
    try:
        args = ['--dry-run'] if dry_run else []
        out = __run_pip_install_command(requirement,
                                        *args,
                                        print_cmd=False,
                                        capture_output=True,
                                        exit_on_error=not dry_run)
        stdout = str(out.stdout).strip()
        stdout_lines = stdout.split('\n')
        dependency = requirement.split(' ')[0]

        if not reduce(lambda aggr, line: aggr | line.startswith('Requirement already satisfied: ' + dependency),
                      stdout_lines, False):
            if dry_run:
                __run_pip_install_command(requirement, print_args=True)
            else:
                print(stdout)
    except RuntimeError:
        __pip_install(requirement)


def __normalize_requirement(requirement: str):
    return requirement.replace('_', '-').lower()


def __find_requirements(requirements_file: str, *dependencies: str, raise_error: bool = True) -> List[str]:
    with open(requirements_file, mode='r', encoding='utf-8') as file:
        requirements = {__normalize_requirement(line.split(' ')[0]): line.strip() for line in file.readlines()}

    if dependencies:
        found_requirements = []

        for dependency in dependencies:
            if __normalize_requirement(dependency) in requirements:
                found_requirements.append(requirements[dependency])
            elif raise_error:
                raise RuntimeError('Dependency "' + dependency + '" not found in requirements file "'
                                   + requirements_file + '"')

        return found_requirements

    return list(requirements.values())


def __install_dependencies(requirements_file: str, *dependencies: str):
    for requirement in __find_requirements(requirements_file, *dependencies):
        __pip_install(requirement, dry_run=True)


def __install_module_dependencies(module: Module, *dependencies: str):
    requirements_file = module.requirements_file

    if path.isfile(requirements_file):
        __install_dependencies(requirements_file, *dependencies)


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


def run_program(program: str,
                *args,
                print_args: bool = False,
                additional_dependencies: Optional[List[str]] = None,
                requirements_file: str = BUILD_MODULE.requirements_file,
                install_program: bool = True,
                env=None):
    """
    Runs an external program that has been installed into the virtual environment.

    :param program:                 The name of the program to be run
    :param args:                    Optional arguments that should be passed to the program
    :param print_args:              True, if the arguments should be included in log statements, False otherwise
    :param additional_dependencies: The names of dependencies that should be installed before running the program
    :param requirements_file:       The path of the requirements.txt file that specifies the dependency versions
    :param install_program:         True, if the program should be installed before being run, False otherwise
    :param env:                     The environment variables to be passed to the program
    """
    dependencies = []

    if install_program:
        dependencies.append(program)

    if additional_dependencies:
        dependencies.extend(additional_dependencies)

    __install_dependencies(requirements_file, *dependencies)
    __run_command(program, *args, print_args=print_args, env=env)


def run_python_program(program: str,
                       *args,
                       print_args: bool = False,
                       additional_dependencies: Optional[List[str]] = None,
                       requirements_file: str = BUILD_MODULE.requirements_file,
                       install_program: bool = True,
                       env=None):
    """
    Runs an external Python program.

    :param program:                 The name of the program to be run
    :param args:                    Optional arguments that should be passed to the program
    :param print_args:              True, if the arguments should be included in log statements, False otherwise
    :param additional_dependencies: The names of dependencies that should be installed before running the program
    :param requirements_file:       The path of the requirements.txt file that specifies the dependency versions
    :param install_program:         True, if the program should be installed before being run, False otherwise
    :param env:                     The environment variable to be passed to the program
    """
    dependencies = []

    if install_program:
        dependencies.append(program)

    if additional_dependencies:
        dependencies.extend(additional_dependencies)

    run_program(path.join(path.dirname(sys.executable), 'python'),
                '-m',
                program,
                *args,
                print_args=print_args,
                additional_dependencies=dependencies,
                requirements_file=requirements_file,
                install_program=False,
                env=env)


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
                if __find_requirements(requirements_file, dependency, raise_error=False):
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
