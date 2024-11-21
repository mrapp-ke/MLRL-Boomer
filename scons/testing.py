"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running automated tests.
"""
from os import environ, path

from modules import CPP_MODULE, PYTHON_MODULE
from util.env import get_env_bool
from util.run import Program, PythonModule


def tests_cpp(**_):
    """
    Runs all automated tests of C++ code.
    """
    Program('meson', 'test', '-C', CPP_MODULE.build_dir, '-v') \
        .print_arguments(True) \
        .run()


def tests_python(**_):
    """
    Runs all automated tests of Python code.
    """
    output_directory = path.join(PYTHON_MODULE.build_dir, 'test-results')
    fail_fast = get_env_bool(environ, 'FAIL_FAST')

    for subproject in PYTHON_MODULE.find_subprojects():
        test_dir = subproject.test_dir

        if path.isdir(test_dir):
            print('Running automated tests for subpackage "' + subproject.name + '"...')
            PythonModule('xmlrunner', 'discover', '--verbose', '--start-directory', test_dir, '--output',
                         output_directory) \
                .add_conditional_arguments(fail_fast, '--failfast') \
                .print_arguments(True) \
                .install_program(False) \
                .add_dependencies('unittest-xml-reporting') \
                .run()
