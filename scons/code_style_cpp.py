"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from modules_old import CPP_MODULE
from util.run import Program


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
