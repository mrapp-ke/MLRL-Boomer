"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from code_style.python.isort import ISort
from code_style.python.pylint import PyLint
from code_style.python.yapf import Yapf
from modules_old import BUILD_MODULE, DOC_MODULE, PYTHON_MODULE


def check_python_code_style(**_):
    """
    Checks if the Python source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE]:
        directory = module.root_dir
        print('Checking Python code style in directory "' + directory + '"...')
        ISort(directory).run()
        Yapf(directory).run()
        PyLint(directory).run()


def enforce_python_code_style(**_):
    """
    Enforces the Python source files to adhere to the code style definitions.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE, DOC_MODULE]:
        directory = module.root_dir
        print('Formatting Python code in directory "' + directory + '"...')
        ISort(directory, enforce_changes=True).run()
        Yapf(directory, enforce_changes=True).run()
