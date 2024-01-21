"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running automated tests.
"""
from os import path

from modules import CPP_MODULE, PYTHON_MODULE
from run import run_program, run_python_program


def __meson_test(build_dir: str):
    run_program('meson', 'test', '-C', build_dir, '-v', print_args=True)


def __python_unittest(directory: str):
    run_python_program('unittest',
                       'discover',
                       '-v',
                       '-f',
                       '-s',
                       directory,
                       install_program=False,
                       additional_dependencies=['unittest-xml-reporting'])


def tests_cpp(**_):
    """
    Runs all automated tests of C++ code.
    """
    __meson_test(CPP_MODULE.build_dir)


def tests_python(**_):
    """
    Runs all automated tests of Python code.
    """
    for subproject in PYTHON_MODULE.find_subprojects():
        test_dir = subproject.test_dir

        if path.isdir(test_dir):
            print('Running automated tests for subpackage "' + subproject.name + '"...')
            __python_unittest(test_dir)
