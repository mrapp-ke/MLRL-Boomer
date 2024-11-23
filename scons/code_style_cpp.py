"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from code_style.cpp.clang_format import ClangFormat
from code_style.cpp.cpplint import CppLint
from modules_old import CPP_MODULE


def check_cpp_code_style(**_):
    """
    Checks if the C++ source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    root_dir = CPP_MODULE.root_dir
    print('Checking C++ code style in directory "' + root_dir + '"...')
    ClangFormat(root_dir).run()

    for subproject in CPP_MODULE.find_subprojects():
        for directory in [subproject.include_dir, subproject.src_dir]:
            CppLint(directory).run()


def enforce_cpp_code_style(**_):
    """
    Enforces the C++ source files to adhere to the code style definitions.
    """
    root_dir = CPP_MODULE.root_dir
    print('Formatting C++ code in directory "' + root_dir + '"...')
    ClangFormat(root_dir, enforce_changes=True).run()
