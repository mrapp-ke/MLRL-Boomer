"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines the individual targets of the build process.
"""
import sys

from functools import reduce
from os import path

from code_style import check_cpp_code_style, check_python_code_style, enforce_cpp_code_style, enforce_python_code_style
from compilation import compile_cpp, compile_cython, install_cpp, install_cython, setup_cpp, setup_cython
from documentation import apidoc_cpp, apidoc_python, doc
from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from packaging import build_python_wheel, install_python_wheels
from run import install_runtime_dependencies
from testing import run_tests

from SCons.Script import COMMAND_LINE_TARGETS
from SCons.Script.SConscript import SConsEnvironment


def __create_phony_target(environment, target, action=None):
    return environment.AlwaysBuild(environment.Alias(target, None, action))


def __print_if_clean(environment, message: str):
    if environment.GetOption('clean'):
        print(message)


# Define target names...
TARGET_NAME_TEST_FORMAT = 'test_format'
TARGET_NAME_TEST_FORMAT_PYTHON = TARGET_NAME_TEST_FORMAT + '_python'
TARGET_NAME_TEST_FORMAT_CPP = TARGET_NAME_TEST_FORMAT + '_cpp'
TARGET_NAME_FORMAT = 'format'
TARGET_NAME_FORMAT_PYTHON = TARGET_NAME_FORMAT + '_python'
TARGET_NAME_FORMAT_CPP = TARGET_NAME_FORMAT + '_cpp'
TARGET_NAME_VENV = 'venv'
TARGET_NAME_COMPILE = 'compile'
TARGET_NAME_COMPILE_CPP = TARGET_NAME_COMPILE + '_cpp'
TARGET_NAME_COMPILE_CYTHON = TARGET_NAME_COMPILE + '_cython'
TARGET_NAME_INSTALL = 'install'
TARGET_NAME_INSTALL_CPP = TARGET_NAME_INSTALL + '_cpp'
TARGET_NAME_INSTALL_CYTHON = TARGET_NAME_INSTALL + '_cython'
TARGET_NAME_BUILD_WHEELS = 'build_wheels'
TARGET_NAME_INSTALL_WHEELS = 'install_wheels'
TARGET_NAME_TESTS = 'tests'
TARGET_NAME_APIDOC = 'apidoc'
TARGET_NAME_APIDOC_CPP = TARGET_NAME_APIDOC + '_cpp'
TARGET_NAME_APIDOC_PYTHON = TARGET_NAME_APIDOC + '_python'
TARGET_NAME_DOC = 'doc'

VALID_TARGETS = {
    TARGET_NAME_TEST_FORMAT, TARGET_NAME_TEST_FORMAT_PYTHON, TARGET_NAME_TEST_FORMAT_CPP, TARGET_NAME_FORMAT,
    TARGET_NAME_FORMAT_PYTHON, TARGET_NAME_FORMAT_CPP, TARGET_NAME_VENV, TARGET_NAME_COMPILE, TARGET_NAME_COMPILE_CPP,
    TARGET_NAME_COMPILE_CYTHON, TARGET_NAME_INSTALL, TARGET_NAME_INSTALL_CPP, TARGET_NAME_INSTALL_CYTHON,
    TARGET_NAME_BUILD_WHEELS, TARGET_NAME_INSTALL_WHEELS, TARGET_NAME_TESTS, TARGET_NAME_APIDOC, TARGET_NAME_APIDOC_CPP,
    TARGET_NAME_APIDOC_PYTHON, TARGET_NAME_DOC
}

DEFAULT_TARGET = TARGET_NAME_INSTALL_WHEELS

# Raise an error if any invalid targets are given...
invalid_targets = [target for target in COMMAND_LINE_TARGETS if target not in VALID_TARGETS]

if invalid_targets:
    print('The following targets are unknown: '
          + reduce(lambda aggr, target: aggr + (', ' if len(aggr) > 0 else '') + target, invalid_targets, ''))
    sys.exit(-1)

# Create temporary file ".sconsign.dblite" in the build directory...
env = SConsEnvironment()
env.SConsignFile(name=path.relpath(path.join(BUILD_MODULE.build_dir, '.sconsign'), BUILD_MODULE.root_dir))

# Define targets for checking code style definitions...
target_test_format_python = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_PYTHON, action=check_python_code_style)
target_test_format_cpp = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_CPP, action=check_cpp_code_style)
target_test_format = __create_phony_target(env, TARGET_NAME_TEST_FORMAT)
env.Depends(target_test_format, [target_test_format_python, target_test_format_cpp])

# Define targets for enforcing code style definitions...
target_format_python = __create_phony_target(env, TARGET_NAME_FORMAT_PYTHON, action=enforce_python_code_style)
target_format_cpp = __create_phony_target(env, TARGET_NAME_FORMAT_CPP, action=enforce_cpp_code_style)
target_format = __create_phony_target(env, TARGET_NAME_FORMAT)
env.Depends(target_format, [target_format_python, target_format_cpp])

# Define target for installing runtime dependencies...
target_venv = __create_phony_target(env, TARGET_NAME_VENV, action=install_runtime_dependencies)

# Define targets for compiling the C++ and Cython code...
env.Command(CPP_MODULE.build_dir, None, action=setup_cpp)
target_compile_cpp = __create_phony_target(env, TARGET_NAME_COMPILE_CPP, action=compile_cpp)
env.Depends(target_compile_cpp, [target_venv, CPP_MODULE.build_dir])

env.Command(PYTHON_MODULE.build_dir, None, action=setup_cython)
target_compile_cython = __create_phony_target(env, TARGET_NAME_COMPILE_CYTHON, action=compile_cython)
env.Depends(target_compile_cython, [target_compile_cpp, PYTHON_MODULE.build_dir])

target_compile = __create_phony_target(env, TARGET_NAME_COMPILE)
env.Depends(target_compile, [target_compile_cpp, target_compile_cython])

# Define targets for cleaning up C++ and Cython build directories...
if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_COMPILE_CPP in COMMAND_LINE_TARGETS \
        or TARGET_NAME_COMPILE in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing C++ build files...')
    env.Clean([target_compile_cpp, DEFAULT_TARGET], CPP_MODULE.build_dir)

if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_COMPILE_CYTHON in COMMAND_LINE_TARGETS \
        or TARGET_NAME_COMPILE in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing Cython build files...')
    env.Clean([target_compile_cython, DEFAULT_TARGET], PYTHON_MODULE.build_dir)

# Define targets for installing shared libraries and extension modules into the source tree...
target_install_cpp = __create_phony_target(env, TARGET_NAME_INSTALL_CPP, action=install_cpp)
env.Depends(target_install_cpp, target_compile_cpp)

target_install_cython = __create_phony_target(env, TARGET_NAME_INSTALL_CYTHON, action=install_cython)
env.Depends(target_install_cython, target_compile_cython)

target_install = env.Alias(TARGET_NAME_INSTALL, None, None)
env.Depends(target_install, [target_install_cpp, target_install_cython])

# Define targets for removing shared libraries and extension modules from the source tree...
if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL_CPP in COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing shared libraries from source tree...')

    for subproject in PYTHON_MODULE.find_subprojects():
        env.Clean([target_install_cpp, DEFAULT_TARGET], subproject.find_shared_libraries())

if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL_CYTHON in COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing extension modules from source tree...')

    for subproject in PYTHON_MODULE.find_subprojects():
        env.Clean([target_install_cython, DEFAULT_TARGET], subproject.find_extension_modules())

# Define targets for building and installing Python wheels...
commands_build_wheels = []
commands_install_wheels = []

for subproject in PYTHON_MODULE.find_subprojects():
    wheels = subproject.find_wheels()
    targets_build_wheels = wheels if wheels else subproject.dist_dir

    command_build_wheels = env.Command(targets_build_wheels, subproject.find_source_files(), action=build_python_wheel)
    commands_build_wheels.append(command_build_wheels)

    command_install_wheels = env.Command(subproject.root_dir, targets_build_wheels, action=install_python_wheels)
    env.Depends(command_install_wheels, command_build_wheels)
    commands_install_wheels.append(command_install_wheels)

target_build_wheels = env.Alias(TARGET_NAME_BUILD_WHEELS, None, None)
env.Depends(target_build_wheels, [target_install] + commands_build_wheels)

target_install_wheels = env.Alias(TARGET_NAME_INSTALL_WHEELS, None, None)
env.Depends(target_install_wheels, [target_install] + commands_install_wheels)

# Define target for cleaning up Python wheels and associated build directories...
if not COMMAND_LINE_TARGETS or TARGET_NAME_BUILD_WHEELS in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing Python wheels...')

    for subproject in PYTHON_MODULE.find_subprojects():
        env.Clean([target_build_wheels, DEFAULT_TARGET], subproject.build_dirs)

# Define targets for running automated tests...
target_test = __create_phony_target(env, TARGET_NAME_TESTS, action=run_tests)
env.Depends(target_test, target_install_wheels)

# Define targets for generating the documentation...
commands_apidoc_cpp = []
commands_apidoc_python = []

for subproject in CPP_MODULE.find_subprojects():
    apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
    config_file = apidoc_subproject.config_file

    if path.isfile(config_file):
        apidoc_files = apidoc_subproject.find_apidoc_files()
        targets_apidoc_cpp = apidoc_files if apidoc_files else apidoc_subproject.apidoc_dir
        source_files = [config_file] + subproject.find_source_files()
        command_apidoc_cpp = env.Command(targets_apidoc_cpp, source_files, action=apidoc_cpp)
        commands_apidoc_cpp.append(command_apidoc_cpp)

target_apidoc_cpp = env.Alias(TARGET_NAME_APIDOC_CPP, None, None)
env.Depends(target_apidoc_cpp, commands_apidoc_cpp)

for subproject in PYTHON_MODULE.find_subprojects():
    apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
    config_file = apidoc_subproject.config_file

    if path.isfile(config_file):
        apidoc_files = apidoc_subproject.find_apidoc_files()
        targets_apidoc_python = apidoc_files if apidoc_files else apidoc_subproject.apidoc_dir
        source_files = [config_file] + subproject.find_source_files()
        command_apidoc_python = env.Command(targets_apidoc_python, source_files, action=apidoc_python)
        env.Depends(command_apidoc_python, target_install_wheels)
        commands_apidoc_python.append(command_apidoc_python)

target_apidoc_python = env.Alias(TARGET_NAME_APIDOC_PYTHON, None, None)
env.Depends(target_apidoc_python, commands_apidoc_python)

target_apidoc = env.Alias(TARGET_NAME_APIDOC, None, None)
env.Depends(target_apidoc, [target_apidoc_cpp, target_apidoc_python])

doc_files = DOC_MODULE.find_build_files()
targets_doc = doc_files if doc_files else DOC_MODULE.build_dir
command_doc = env.Command(targets_doc, [DOC_MODULE.config_file] + DOC_MODULE.find_source_files(), action=doc)
env.Depends(command_doc, target_apidoc)
target_doc = env.Alias(TARGET_NAME_DOC, None, None)
env.Depends(target_doc, command_doc)

# Define target for cleaning up the documentation and associated build directories...
if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC_CPP in COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC in COMMAND_LINE_TARGETS \
        or TARGET_NAME_DOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing C++ API documentation...')

    for subproject in CPP_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
        env.Clean([target_apidoc_cpp, DEFAULT_TARGET], apidoc_subproject.find_build_files())

if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC_PYTHON in COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC in COMMAND_LINE_TARGETS \
        or TARGET_NAME_DOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing Python API documentation...')

    for subproject in PYTHON_MODULE.find_subprojects():
        apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
        env.Clean([target_apidoc_python, DEFAULT_TARGET], apidoc_subproject.find_build_files())

if not COMMAND_LINE_TARGETS or TARGET_NAME_APIDOC in COMMAND_LINE_TARGETS or TARGET_NAME_DOC in COMMAND_LINE_TARGETS:
    env.Clean([target_apidoc, DEFAULT_TARGET], DOC_MODULE.apidoc_dir)

if not COMMAND_LINE_TARGETS or TARGET_NAME_DOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing documentation...')
    env.Clean([target_doc, DEFAULT_TARGET], DOC_MODULE.build_dir)

# Set the default target...
env.Default(DEFAULT_TARGET)