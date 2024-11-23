"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from modules import DOC_MODULE, PYTHON_MODULE
from util.run import Program

MD_DIRS = [('.', False), (DOC_MODULE.root_dir, True), (PYTHON_MODULE.root_dir, True)]


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
