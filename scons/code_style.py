"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
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


def __clang_format(directory: str, enforce_changes: bool = False):
    cpp_header_files = glob(path.join(directory, '**', '*.hpp'), recursive=True)
    cpp_source_files = glob(path.join(directory, '**', '*.cpp'), recursive=True)
    args = ['--style=file']

    if enforce_changes:
        args.append('-i')
    else:
        args.append('-n')
        args.append('--Werror')

    run_program('clang-format', *args, *cpp_header_files, *cpp_source_files)


def __mdformat(directory: str, recursive: bool = False, enforce_changes: bool = False):
    md_files = glob(path.join(directory, '**', '*.md') if recursive else path.join(directory, '*.md'))
    args = ['--number', '--wrap', 'no', '--end-of-line', 'lf']

    if not enforce_changes:
        args.append('--check')

    run_program('mdformat', *args, *md_files, additional_dependencies=['mdformat-myst'])


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
    for module in [BUILD_MODULE, PYTHON_MODULE, DOC_MODULE]:
        directory = module.root_dir
        print('Formatting Python code in directory "' + directory + '"...')
        __isort(directory, enforce_changes=True)
        __yapf(directory, enforce_changes=True)


def check_cpp_code_style(**_):
    """
    Checks if the C++ source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    directory = CPP_MODULE.root_dir
    print('Checking C++ code style in directory "' + directory + '"...')
    __clang_format(directory)


def enforce_cpp_code_style(**_):
    """
    Enforces the C++ source files to adhere to the code style definitions.
    """
    directory = CPP_MODULE.root_dir
    print('Formatting C++ code in directory "' + directory + '"...')
    __clang_format(directory, enforce_changes=True)


def check_md_code_style(**_):
    """
    Check if the Markdown files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    print('Checking Markdown code style in the root directory...')
    __mdformat('.')
    directory = DOC_MODULE.root_dir
    print('Checking Markdown code style in the directory "' + directory + '"...')
    __mdformat(directory, recursive=True)


def enforce_md_code_style(**_):
    """
    Enforces the Markdown files to adhere to the code style definitions.
    """
    print('Formatting Markdown files in the root directory...')
    __mdformat('.', enforce_changes=True)
    directory = DOC_MODULE.root_dir
    print('Formatting Markdown files in the directory "' + directory + '"...')
    __mdformat(directory, recursive=True, enforce_changes=True)
