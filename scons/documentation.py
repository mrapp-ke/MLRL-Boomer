"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for generating the documentation.
"""
from os import environ, makedirs, path, remove
from typing import List

from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from util.env import set_env
from util.io import read_file, write_file
from util.pip import RequirementsFile
from util.run import Program


def __doxygen(project_name: str, input_dir: str, output_dir: str):
    makedirs(output_dir, exist_ok=True)
    env = environ.copy()
    set_env(env, 'DOXYGEN_PROJECT_NAME', 'libmlrl' + project_name)
    set_env(env, 'DOXYGEN_INPUT_DIR', input_dir)
    set_env(env, 'DOXYGEN_OUTPUT_DIR', output_dir)
    set_env(env, 'DOXYGEN_PREDEFINED', 'MLRL' + project_name.upper() + '_API=')
    Program(RequirementsFile(BUILD_MODULE.requirements_file), 'doxygen', DOC_MODULE.doxygen_config_file) \
        .print_arguments(False) \
        .install_program(False) \
        .use_environment(env) \
        .run()


def __breathe_apidoc(source_dir: str, output_dir: str, project: str):
    Program(RequirementsFile(DOC_MODULE.requirements_file), 'breathe-apidoc', '--members', '--project', project, '-g',
            'file', '-o', output_dir, source_dir) \
        .print_arguments(True) \
        .add_dependencies('breathe') \
        .install_program(False) \
        .run()


def __sphinx_apidoc(source_dir: str, output_dir: str):
    Program(RequirementsFile(DOC_MODULE.requirements_file), 'sphinx-apidoc', '--separate', '--module-first', '--no-toc',
            '-o', output_dir, source_dir, '*.so*') \
        .print_arguments(True) \
        .add_dependencies('sphinx') \
        .install_program(False) \
        .run()

    root_rst_file = path.join(output_dir, 'mlrl.rst')

    if path.isfile(root_rst_file):
        remove(root_rst_file)


def __sphinx_build(source_dir: str, output_dir: str):
    Program(RequirementsFile(DOC_MODULE.requirements_file), 'sphinx-build', '--jobs', 'auto', source_dir, output_dir) \
        .print_arguments(True) \
        .add_dependencies('furo', 'myst-parser', 'sphinxext-opengraph', 'sphinx-inline-tabs', 'sphinx-copybutton',
                          'sphinx-favicon',) \
        .install_program(False) \
        .run()


def __read_tocfile_template(directory: str) -> List[str]:
    with read_file(path.join(directory, 'index.md.template')) as file:
        return file.readlines()


def __write_tocfile(directory: str, tocfile_entries: List[str]):
    tocfile_template = __read_tocfile_template(directory)
    tocfile = []

    for line in tocfile_template:
        if line.strip() == '%s':
            tocfile.extend(tocfile_entries)
        else:
            tocfile.append(line)

    with write_file(path.join(directory, 'index.md')) as file:
        file.writelines(tocfile)


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

        if apidoc_subproject:
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
    tocfile_entries = []

    for subproject in CPP_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
        root_file = apidoc_subproject.root_file

        if path.isfile(root_file):
            tocfile_entries.append('Library libmlrl' + apidoc_subproject.name + ' <'
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

        if apidoc_subproject:
            print('Generating Python API documentation for subproject "' + apidoc_subproject.name + '"...')
            build_dir = apidoc_subproject.build_dir
            makedirs(build_dir, exist_ok=True)
            __sphinx_apidoc(source_dir=apidoc_subproject.source_subproject.source_dir, output_dir=build_dir)


def apidoc_python_tocfile(**_):
    """
    Generates a tocfile referencing the Python API documentation for all existing subprojects.
    """
    print('Generating tocfile referencing the Python API documentation for all subprojects...')
    tocfile_entries = []

    for subproject in PYTHON_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
        root_file = apidoc_subproject.root_file

        if path.isfile(root_file):
            tocfile_entries.append('Package mlrl-' + apidoc_subproject.name + ' <'
                                   + path.relpath(root_file, DOC_MODULE.apidoc_dir_python) + '>\n')

    __write_tocfile(DOC_MODULE.apidoc_dir_python, tocfile_entries)


def doc(**_):
    """
    Builds the documentation.
    """
    print('Generating documentation...')
    __sphinx_build(source_dir=DOC_MODULE.root_dir, output_dir=DOC_MODULE.build_dir)
