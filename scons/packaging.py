"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for building and installing Python wheel packages.
"""
from typing import List

from modules import PYTHON_MODULE
from run import run_python_program


def __build_python_wheel(package_dir: str):
    run_python_program('build',
                       '--no-isolation',
                       '--wheel',
                       package_dir,
                       print_args=True,
                       additional_dependencies=['wheel', 'setuptools'])


def __install_python_wheels(wheels: List[str]):
    run_python_program('pip',
                       'install',
                       '--force-reinstall',
                       '--no-deps',
                       '--disable-pip-version-check',
                       *wheels,
                       print_args=True,
                       install_program=False)


# pylint: disable=unused-argument
def build_python_wheel(env, target, source):
    """
    Builds a Python wheel package for a single subproject.

    :param env:     The scons environment
    :param target:  The path of the wheel package to be built, if it does already exist, or the path of the directory,
                    where the wheel package should be stored
    :param source:  The source files from which the wheel package should be built
    """
    if target:
        subproject = PYTHON_MODULE.find_subproject(target[0].path)
        print('Building Python wheels for subproject "' + subproject.name + '"...')
        __build_python_wheel(subproject.root_dir)


# pylint: disable=unused-argument
def install_python_wheels(env, target, source):
    """
    Installs all Python wheel packages that have been built for a single subproject.

    :param env:     The scons environment
    :param target:  The path of the subproject's root directory
    :param source:  The paths of the wheel packages to be installed
    """
    if source:
        subproject = PYTHON_MODULE.find_subproject(source[0].path)
        print('Installing Python wheels for subproject "' + subproject.name + '"...')
        __install_python_wheels(subproject.find_wheels())
