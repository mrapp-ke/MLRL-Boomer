"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from modules import BUILD_MODULE, PYTHON_MODULE
from run import run_program


def __isort(directory: str, enforce_changes: bool = False):
    args = ['--settings-path', '.', '--virtual-env', 'venv', '--skip-gitignore']

    if not enforce_changes:
        args.append('--check')

    run_program('isort', *args, directory)


def __yapf(directory: str, enforce_changes: bool = False):
    args = ['-r', '-p', '--style=.style.yapf', '--exclude', '**/build/*.py', '-i' if enforce_changes else '--diff']
    run_program('yapf', *args, directory)


def __pylint(directory: str):
    args = ['--jobs=0', '--recursive=y', '--ignore=build', '--rcfile=.pylintrc', '--score=n']
    run_program('pylint', *args, directory)


def check_python_code_style(**_):
    """
    Checks if the Python source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE]:
        directory = module.root_dir
        print('Checking Python code style in directory "' + directory + '"...')
        __isort(directory)
        __yapf(directory)
        __pylint(directory)


def enforce_python_code_style(**_):
    """
    Enforces the Python source files to adhere to the code style definitions.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE]:
        directory = module.root_dir
        print('Formatting Python code in directory "' + directory + '"...')
        __isort(directory, enforce_changes=True)
        __yapf(directory, enforce_changes=True)
