"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running external programs during the build process.
"""
from typing import List, Optional

from dependencies import install_dependencies
from modules import BUILD_MODULE
from util.cmd import Command


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

    install_dependencies(requirements_file, *dependencies)
    Command(program, *args).print_arguments(print_args).use_environment(env).run()


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

    run_program('python',
                '-m',
                program,
                *args,
                print_args=print_args,
                additional_dependencies=dependencies,
                requirements_file=requirements_file,
                install_program=False,
                env=env)
