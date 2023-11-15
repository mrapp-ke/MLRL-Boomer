"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for generating the documentation.
"""
from os import makedirs, path, remove
from typing import List

from environment import set_env, unset_env
from modules import CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from run import run_program

ENV_DOXYGEN_PROJECT_NAME = 'DOXYGEN_PROJECT_NAME'

ENV_DOXYGEN_INPUT_DIR = 'DOXYGEN_INPUT_DIR'

ENV_DOXYGEN_OUTPUT_DIR = 'DOXYGEN_OUTPUT_DIR'

ENV_DOXYGEN_PREDEFINED = 'DOXYGEN_PREDEFINED'


def __doxygen(project_name: str, input_dir: str, output_dir: str):
    makedirs(output_dir, exist_ok=True)
    set_env(ENV_DOXYGEN_PROJECT_NAME, 'libmlrl' + project_name)
    set_env(ENV_DOXYGEN_INPUT_DIR, input_dir)
    set_env(ENV_DOXYGEN_OUTPUT_DIR, output_dir)
    set_env(ENV_DOXYGEN_PREDEFINED, 'MLRL' + project_name.upper() + '_API=')
    run_program('doxygen', DOC_MODULE.doxygen_config_file, print_args=True)
    unset_env(ENV_DOXYGEN_PROJECT_NAME)
    unset_env(ENV_DOXYGEN_INPUT_DIR)
    unset_env(ENV_DOXYGEN_OUTPUT_DIR)
    unset_env(ENV_DOXYGEN_PREDEFINED)


def __breathe_apidoc(source_dir: str, output_dir: str, project: str):
    run_program('breathe-apidoc',
                '--members',
                '--project',
                project,
                '-g',
                'file',
                '-o',
                output_dir,
                source_dir,
                print_args=True,
                additional_dependencies=['breathe'],
                requirements_file=DOC_MODULE.requirements_file)


def __sphinx_apidoc(source_dir: str, output_dir: str):
    run_program('sphinx-apidoc',
                '--separate',
                '--module-first',
                '--no-toc',
                '-o',
                output_dir,
                source_dir,
                '*.so*',
                print_args=True,
                additional_dependencies=['sphinx'],
                requirements_file=DOC_MODULE.requirements_file)

    root_rst_file = path.join(output_dir, 'mlrl.rst')

    if path.isfile(root_rst_file):
        remove(root_rst_file)


def __sphinx_build(source_dir: str, output_dir: str):
    run_program('sphinx-build',
                '--jobs',
                'auto',
                source_dir,
                output_dir,
                print_args=True,
                additional_dependencies=[
                    'furo', 'sphinxext-opengraph', 'sphinx-inline-tabs', 'sphinx-copybutton', 'myst-parser'
                ],
                requirements_file=DOC_MODULE.requirements_file)


def __read_tocfile_template(dir: str) -> List[str]:
    with open(path.join(dir, 'index.rst.template'), mode='r', encoding='utf-8') as file:
        return file.readlines()


def __write_tocfile(dir: str, tocfile_entries: List[str]):
    tocfile_template = __read_tocfile_template(dir)

    with open(path.join(dir, 'index.rst'), mode='w', encoding='utf-8') as file:
        file.writelines(tocfile_template)
        file.writelines(tocfile_entries)


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
        subproject_name = apidoc_subproject.name
        print('Generating C++ API documentation for subproject "' + subproject_name + '"...')
        include_dir = path.join(apidoc_subproject.source_subproject.root_dir, 'include')
        build_dir = apidoc_subproject.build_dir
        __doxygen(project_name=subproject_name, input_dir=include_dir, output_dir=build_dir)
        __breathe_apidoc(source_dir=path.join(build_dir, 'xml'), output_dir=build_dir, project=subproject_name)


def apidoc_cpp_tocfile(**_):
    """
    Generates a tocfile referencing the C++ API documentation for all existing subprojects.
    """
    print('Generating tocfile referencing the C++ API documentation for all subprojects...')
    tocfile_entries = ['\n']

    for subproject in CPP_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
        root_file = apidoc_subproject.root_file

        if path.isfile(root_file):
            tocfile_entries.append('    Library libmlrl' + apidoc_subproject.name + ' <'
                                   + path.relpath(root_file, DOC_MODULE.apidoc_dir_cpp) + '>\n')

    __write_tocfile(DOC_MODULE.apidoc_dir_cpp, tocfile_entries)


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
        print('Generating Python API documentation for subproject "' + apidoc_subproject.name + '"...')
        build_dir = apidoc_subproject.build_dir
        makedirs(build_dir, exist_ok=True)
        __sphinx_apidoc(source_dir=apidoc_subproject.source_subproject.source_dir, output_dir=build_dir)


def apidoc_python_tocfile(**_):
    """
    Generates a tocfile referencing the Python API documentation for all existing subprojects.
    """
    print('Generating tocfile referencing the Python API documentation for all subprojects...')
    tocfile_entries = ['\n']

    for subproject in PYTHON_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
        root_file = apidoc_subproject.root_file

        if path.isfile(root_file):
            tocfile_entries.append('    Package mlrl-' + apidoc_subproject.name + ' <'
                                   + path.relpath(root_file, DOC_MODULE.apidoc_dir_python) + '>\n')

    __write_tocfile(DOC_MODULE.apidoc_dir_python, tocfile_entries)


def doc(**_):
    """
    Builds the documentation.
    """
    print('Generating documentation...')
    __sphinx_build(source_dir=DOC_MODULE.root_dir, output_dir=DOC_MODULE.build_dir)
