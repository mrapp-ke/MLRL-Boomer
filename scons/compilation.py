"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from typing import List, Optional

from modules import CPP_MODULE, PYTHON_MODULE
from run import run_program


def __meson_setup(root_dir: str, build_dir: str, dependencies: Optional[List[str]] = None):
    print('Setting up build directory "' + build_dir + '"...')
    run_program('meson', 'setup', build_dir, root_dir, print_args=True, additional_dependencies=dependencies)


def __meson_compile(build_dir: str):
    run_program('meson', 'compile', '-C', build_dir, print_args=True)


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    __meson_setup(CPP_MODULE.root_dir, CPP_MODULE.build_dir, dependencies=['ninja'])


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    print('Compiling C++ code...')
    __meson_compile(CPP_MODULE.build_dir)


def setup_cython(**_):
    """
    Sets up the build system for compiling the Cython code.
    """
    __meson_setup(PYTHON_MODULE.root_dir, PYTHON_MODULE.build_dir, dependencies=['cython'])


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    print('Compiling Cython code...')
    __meson_compile(PYTHON_MODULE.build_dir)
