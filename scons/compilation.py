"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from typing import List, Optional

from modules import CPP_MODULE, PYTHON_MODULE
from run import run_venv_program


def __meson_setup(root_dir: str, build_dir: str, dependencies: Optional[List[str]] = None):
    print('Setting up build directory "' + build_dir + '"...')
    run_venv_program('meson', 'setup', build_dir, root_dir, print_args=True, additional_dependencies=dependencies)


def __meson_compile(build_dir: str):
    run_venv_program('meson', 'compile', '-C', build_dir, print_args=True)


def __meson_install(build_dir: str):
    run_venv_program('meson', 'install', '--no-rebuild', '--only-changed', '-C', build_dir, print_args=True)


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


def install_cpp(**_):
    """
    Installs shared libraries into the source tree.
    """
    print('Installing shared libraries into source tree...')
    __meson_install(CPP_MODULE.build_dir)


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


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    __meson_install(PYTHON_MODULE.build_dir)