"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from run import run_program

MD_DIRS = [('.', False), (DOC_MODULE.root_dir, True), (PYTHON_MODULE.root_dir, True)]

YAML_DIRS = [('.', False), ('.github', True)]


def __isort(directory: str, enforce_changes: bool = False):
    args = ['--settings-path', '.', '--virtual-env', 'venv', '--skip-gitignore']

    if not enforce_changes:
        args.append('--check')

    run_program('isort', *args, directory)


def __yapf(directory: str, enforce_changes: bool = False):
    run_program('yapf', '-r', '-p', '--style=.style.yapf', '--exclude', '**/build/*.py',
                '-i' if enforce_changes else '--diff', directory)


def __pylint(directory: str):
    run_program('pylint', '--jobs=0', '--recursive=y', '--ignore=build', '--rcfile=.pylintrc', '--score=n', directory)


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


def __cpplint(directory: str):
    run_program('cpplint', '--quiet', '--recursive', directory)


def __mdformat(directory: str, recursive: bool = False, enforce_changes: bool = False):
    suffix_md = '*.md'
    glob_path = path.join(directory, '**', '**', suffix_md) if recursive else path.join(directory, suffix_md)
    md_files = glob(glob_path, recursive=recursive)
    args = ['--number', '--wrap', 'no', '--end-of-line', 'lf']

    if not enforce_changes:
        args.append('--check')

    run_program('mdformat', *args, *md_files, additional_dependencies=['mdformat-myst'])


def __yamlfix(directory: str, recursive: bool = False, enforce_changes: bool = False):
    glob_path = path.join(directory, '**', '*') if recursive else path.join(directory, '*')
    glob_path_hidden = path.join(directory, '**', '.*') if recursive else path.join(directory, '.*')
    yaml_files = [
        file for file in glob(glob_path) + glob(glob_path_hidden)
        if path.basename(file).endswith('.yml') or path.basename(file).endswith('.yaml')
    ]
    args = ['--config-file', '.yamlfix.toml']

    if not enforce_changes:
        args.append('--check')

    run_program('yamlfix', *args, *yaml_files, print_args=True)


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
    root_dir = CPP_MODULE.root_dir
    print('Checking C++ code style in directory "' + root_dir + '"...')
    __clang_format(root_dir)

    for subproject in CPP_MODULE.find_subprojects():
        for directory in [subproject.include_dir, subproject.src_dir]:
            __cpplint(directory)


def enforce_cpp_code_style(**_):
    """
    Enforces the C++ source files to adhere to the code style definitions.
    """
    root_dir = CPP_MODULE.root_dir
    print('Formatting C++ code in directory "' + root_dir + '"...')
    __clang_format(root_dir, enforce_changes=True)


def check_md_code_style(**_):
    """
    Checks if the Markdown files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in MD_DIRS:
        print('Checking Markdown code style in the directory "' + directory + '"...')
        __mdformat(directory, recursive=recursive)


def enforce_md_code_style(**_):
    """
    Enforces the Markdown files to adhere to the code style definitions.
    """
    for directory, recursive in MD_DIRS:
        print('Formatting Markdown files in the directory "' + directory + '"...')
        __mdformat(directory, recursive=recursive, enforce_changes=True)


def check_yaml_code_style(**_):
    """
    Checks if the YAML files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in YAML_DIRS:
        print('Checking YAML files in the directory "' + directory + '"...')
        __yamlfix(directory, recursive=recursive)


def enforce_yaml_code_style(**_):
    """
    Enforces the YAML files to adhere to the code style definitions.
    """
    for directory, recursive in YAML_DIRS:
        print('Formatting YAML files in the directory "' + directory + '"...')
        __yamlfix(directory, recursive=recursive, enforce_changes=True)
