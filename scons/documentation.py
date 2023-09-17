"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for generating the documentation.
"""
from os import makedirs, path
from typing import List, Optional

from modules import DOC_MODULE
from run import run_program, run_venv_program


def __doxygen(config_file: str, output_dir: str):
    makedirs(output_dir, exist_ok=True)
    run_program('doxygen', config_file, print_args=True)


def __sphinx_apidoc(source_dir: str, output_dir: str):
    run_venv_program('sphinx-apidoc',
                     '--tocfile',
                     'index',
                     '-f',
                     '-o',
                     output_dir,
                     source_dir,
                     '**/cython',
                     print_args=True,
                     additional_dependencies=['sphinx', 'furo'],
                     requirements_file=DOC_MODULE.requirements_file)


def __sphinx_build(source_dir: str, output_dir: str, additional_dependencies: Optional[List[str]] = None):
    run_venv_program('sphinx-build',
                     '-M',
                     'html',
                     source_dir,
                     output_dir,
                     print_args=True,
                     additional_dependencies=additional_dependencies,
                     requirements_file=DOC_MODULE.requirements_file)


# pylint: disable=unused-argument
def apidoc_cpp(env, target, source):
    """
    Builds the API documentation for a single C++ subproject.

    :param env:     The scons environment
    :param target:  The path of the files that belong to the API documentation, if it has already been built, or the
                    path of the directory, where the API documentation should be stored
    :param source:  The paths of the source files from which the API documentation should be built
    """
    if target:
        apidoc_subproject = DOC_MODULE.find_cpp_apidoc_subproject(target[0].path)
        config_file = apidoc_subproject.config_file

        if path.isfile(config_file):
            print('Generating C++ API documentation for subproject "' + apidoc_subproject.name + '"...')
            __doxygen(config_file=config_file, output_dir=path.join(apidoc_subproject.apidoc_dir))


# pylint: disable=unused-argument
def apidoc_python(env, target, source):
    """
    Builds the API documentation for a single Python subproject.

    :param env:     The scons environment
    :param target:  The path of the files that belong to the API documentation, if it has already been built, or the
                    path of the directory, where the API documentation should be stored
    :param source:  The paths of the source files from which the API documentation should be built
    """
    if target:
        apidoc_subproject = DOC_MODULE.find_python_apidoc_subproject(target[0].path)
        tmp_dir = apidoc_subproject.build_dir

        if path.isdir(tmp_dir):
            print('Generating Python API documentation for subproject "' + apidoc_subproject.name + '"...')
            __sphinx_apidoc(source_dir=apidoc_subproject.source_subproject.source_dir, output_dir=tmp_dir)
            __sphinx_build(source_dir=tmp_dir, output_dir=apidoc_subproject.apidoc_dir)


def doc(**_):
    """
    Builds the documentation.
    """
    print('Generating documentation...')
    __sphinx_build(
        source_dir=DOC_MODULE.root_dir,
        output_dir=DOC_MODULE.build_dir,
        additional_dependencies=['sphinxext-opengraph', 'sphinx-inline-tabs', 'sphinx-copybutton', 'myst-parser'])
