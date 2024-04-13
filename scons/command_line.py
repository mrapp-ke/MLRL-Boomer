"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running command line programs during the build process.
"""
import subprocess
import sys

from functools import reduce
from os import path


def __format_command(cmd: str, *args, format_args: bool = True) -> str:
    return cmd + (reduce(lambda aggr, argument: aggr + ' ' + argument, args, '') if format_args else '')


def __is_virtual_environment() -> bool:
    return sys.prefix != sys.base_prefix


def __get_qualified_command(cmd: str) -> str:
    if __is_virtual_environment():
        # On Windows, we use the relative path to the command's executable within the virtual environment, if such an
        # executable exists. This circumvents situations where the PATH environment variable has not been updated after
        # activating the virtual environment. This can prevent the executables from being found or can lead to the wrong
        # executable, from outside the virtual environment, being executed.
        executable = path.join(sys.prefix, 'Scripts', cmd + '.exe')

        if path.isfile(executable):
            return executable

    return cmd


def run_command(cmd: str,
                *args,
                print_cmd: bool = True,
                print_args: bool = False,
                capture_output: bool = False,
                exit_on_error: bool = True,
                env=None):
    """
    Runs a command line program.

    :param cmd:             The name of the program to be run
    :param args:            Optional arguments that should be passed to the program
    :param print_cmd:       True, if the name of the program should be included in log statements, False otherwise
    :param print_args:      True, if the arguments should be included in log statements, False otherwise
    :param capture_output:  True, if the output of the program should be captured and returned, False otherwise
    :param exit_on_error:   True, if the build system should be terminated when an error occurs, False otherwise
    :param env:             The environment variables to be passed to the program
    """
    cmd = __get_qualified_command(cmd)

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
