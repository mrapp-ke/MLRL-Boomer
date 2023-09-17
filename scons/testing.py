"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running automated tests.
"""
from os import path

from modules import PYTHON_MODULE
from run import run_python_program


def __run_python_tests(directory: str):
    run_python_program('unittest', 'discover', '-v', '-f', '-s', directory)


def run_tests(**_):
    """
    Runs all automated tests.
    """
    for subproject in PYTHON_MODULE.find_subprojects():
        test_dir = subproject.test_dir

        if path.isdir(test_dir):
            print('Running automated tests for subpackage "' + subproject.name + '"...')
            __run_python_tests(test_dir)
