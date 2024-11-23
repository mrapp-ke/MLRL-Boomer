"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from code_style.markdown.mdformat import MdFormat
from modules_old import DOC_MODULE, PYTHON_MODULE

MD_DIRS = [('.', False), (DOC_MODULE.root_dir, True), (PYTHON_MODULE.root_dir, True)]


def check_md_code_style(**_):
    """
    Checks if the Markdown files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in MD_DIRS:
        print('Checking Markdown code style in the directory "' + directory + '"...')
        MdFormat(directory, recursive=recursive).run()


def enforce_md_code_style(**_):
    """
    Enforces the Markdown files to adhere to the code style definitions.
    """
    for directory, recursive in MD_DIRS:
        print('Formatting Markdown files in the directory "' + directory + '"...')
        MdFormat(directory, recursive=recursive, enforce_changes=True).run()
