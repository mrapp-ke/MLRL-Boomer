"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from util.run import Program

MD_DIRS = [('.', False), (DOC_MODULE.root_dir, True), (PYTHON_MODULE.root_dir, True)]

YAML_DIRS = [('.', False), ('.github', True)]


class Yapf(Program):
    """
    Allows to run the external program "yapf".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yapf', '-r', '-p', '--style=.style.yapf', '--exclude', '**/build/*.py',
                         '-i' if enforce_changes else '--diff', directory)


class Isort(Program):
    """
    Allows to run the external program "isort".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('isort', directory, '--settings-path', '.', '--virtual-env', 'venv', '--skip-gitignore')
        self.add_conditional_arguments(not enforce_changes, '--check')


class Pylint(Program):
    """
    Allows to run the external program "pylint".
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory, the program should be applied to
        """
        super().__init__('pylint', directory, '--jobs=0', '--recursive=y', '--ignore=build', '--rcfile=.pylintrc',
                         '--score=n')


class ClangFormat(Program):
    """
    Allows to run the external program "clang-format".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('clang-format', '--style=file')
        self.add_conditional_arguments(enforce_changes, '-i')
        self.add_conditional_arguments(not enforce_changes, '--dry-run', '--Werror')
        self.add_arguments(*glob(path.join(directory, '**', '*.hpp'), recursive=True))
        self.add_arguments(*glob(path.join(directory, '**', '*.cpp'), recursive=True))


class Cpplint(Program):
    """
    Allows to run the external program "cpplint".
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory, the program should be applied to
        """
        super().__init__('cpplint', directory, '--quiet', '--recursive')


class Mdformat(Program):
    """
    Allows to run the external program "mdformat".
    """

    def __init__(self, directory: str, recursive: bool = False, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param recursive:       True, if the program should be applied to subdirectories, False otherwise
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('mdformat', '--number', '--wrap', 'no', '--end-of-line', 'lf')
        self.add_conditional_arguments(not enforce_changes, '--check')
        suffix_md = '*.md'
        glob_path = path.join(directory, '**', '**', suffix_md) if recursive else path.join(directory, suffix_md)
        self.add_arguments(*glob(glob_path, recursive=recursive))
        self.add_dependencies('mdformat-myst')


class Yamlfix(Program):
    """
    Allows to run the external program "yamlfix".
    """

    def __init__(self, directory: str, recursive: bool = False, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param recursive:       True, if the program should be applied to subdirectories, False otherwise
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yamlfix', '--config-file', '.yamlfix.toml')
        self.add_conditional_arguments(not enforce_changes, '--check')
        glob_path = path.join(directory, '**', '*') if recursive else path.join(directory, '*')
        glob_path_hidden = path.join(directory, '**', '.*') if recursive else path.join(directory, '.*')
        yaml_files = [
            file for file in glob(glob_path) + glob(glob_path_hidden)
            if path.basename(file).endswith('.yml') or path.basename(file).endswith('.yaml')
        ]
        self.add_arguments(yaml_files)
        self.print_arguments(True)


def check_python_code_style(**_):
    """
    Checks if the Python source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE]:
        directory = module.root_dir
        print('Checking Python code style in directory "' + directory + '"...')
        Isort(directory).run()
        Yapf(directory).run()
        Pylint(directory).run()


def enforce_python_code_style(**_):
    """
    Enforces the Python source files to adhere to the code style definitions.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE, DOC_MODULE]:
        directory = module.root_dir
        print('Formatting Python code in directory "' + directory + '"...')
        Isort(directory, enforce_changes=True).run()
        Yapf(directory, enforce_changes=True).run()


def check_cpp_code_style(**_):
    """
    Checks if the C++ source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    root_dir = CPP_MODULE.root_dir
    print('Checking C++ code style in directory "' + root_dir + '"...')
    ClangFormat(root_dir).run()

    for subproject in CPP_MODULE.find_subprojects():
        for directory in [subproject.include_dir, subproject.src_dir]:
            Cpplint(directory).run()


def enforce_cpp_code_style(**_):
    """
    Enforces the C++ source files to adhere to the code style definitions.
    """
    root_dir = CPP_MODULE.root_dir
    print('Formatting C++ code in directory "' + root_dir + '"...')
    ClangFormat(root_dir, enforce_changes=True).run()


def check_md_code_style(**_):
    """
    Checks if the Markdown files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in MD_DIRS:
        print('Checking Markdown code style in the directory "' + directory + '"...')
        Mdformat(directory, recursive=recursive).run()


def enforce_md_code_style(**_):
    """
    Enforces the Markdown files to adhere to the code style definitions.
    """
    for directory, recursive in MD_DIRS:
        print('Formatting Markdown files in the directory "' + directory + '"...')
        Mdformat(directory, recursive=recursive, enforce_changes=True).run()


def check_yaml_code_style(**_):
    """
    Checks if the YAML files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in YAML_DIRS:
        print('Checking YAML files in the directory "' + directory + '"...')
        Yamlfix(directory, recursive=recursive).run()


def enforce_yaml_code_style(**_):
    """
    Enforces the YAML files to adhere to the code style definitions.
    """
    for directory, recursive in YAML_DIRS:
        print('Formatting YAML files in the directory "' + directory + '"...')
        Yamlfix(directory, recursive=recursive, enforce_changes=True).run()
