"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines the individual targets of the build process.
"""
import sys

from functools import reduce
from os import path

from changelog import print_latest_changelog, update_changelog_bugfix, update_changelog_feature, \
    update_changelog_main, validate_changelog_bugfix, validate_changelog_feature, validate_changelog_main
from code_style import check_cpp_code_style, check_md_code_style, check_python_code_style, check_yaml_code_style, \
    enforce_cpp_code_style, enforce_md_code_style, enforce_python_code_style, enforce_yaml_code_style
from compilation import compile_cpp, compile_cython, install_cpp, install_cython, setup_cpp, setup_cython
from dependencies import check_dependency_versions, install_runtime_dependencies
from documentation import apidoc_cpp, apidoc_cpp_tocfile, apidoc_python, apidoc_python_tocfile, doc
from github_actions import check_github_actions, update_github_actions
from modules import BUILD_MODULE, CPP_MODULE, DOC_MODULE, PYTHON_MODULE
from packaging import build_python_wheel, install_python_wheels
from testing import tests_cpp, tests_python
from versioning import apply_development_version, increment_development_version, increment_major_version, \
    increment_minor_version, increment_patch_version, print_current_version, reset_development_version

from SCons.Script import COMMAND_LINE_TARGETS
from SCons.Script.SConscript import SConsEnvironment


def __create_phony_target(environment, target, action=None):
    return environment.AlwaysBuild(environment.Alias(target, None, action))


def __print_if_clean(environment, message: str):
    if environment.GetOption('clean'):
        print(message)


# Define target names...
TARGET_NAME_INCREMENT_DEVELOPMENT_VERSION = 'increment_development_version'
TARGET_NAME_RESET_DEVELOPMENT_VERSION = 'reset_development_version'
TARGET_NAME_APPLY_DEVELOPMENT_VERSION = 'apply_development_version'
TARGET_NAME_INCREMENT_PATCH_VERSION = 'increment_patch_version'
TARGET_NAME_INCREMENT_MINOR_VERSION = 'increment_minor_version'
TARGET_NAME_INCREMENT_MAJOR_VERSION = 'increment_major_version'
TARGET_NAME_VALIDATE_CHANGELOG_BUGFIX = 'validate_changelog_bugfix'
TARGET_NAME_VALIDATE_CHANGELOG_FEATURE = 'validate_changelog_feature'
TARGET_NAME_VALIDATE_CHANGELOG_MAIN = 'validate_changelog_main'
TARGET_NAME_UPDATE_CHANGELOG_BUGFIX = 'update_changelog_bugfix'
TARGET_NAME_UPDATE_CHANGELOG_FEATURE = 'update_changelog_feature'
TARGET_NAME_UPDATE_CHANGELOG_MAIN = 'update_changelog_main'
TARGET_NAME_PRINT_VERSION = 'print_version'
TARGET_NAME_PRINT_LATEST_CHANGELOG = 'print_latest_changelog'
TARGET_NAME_TEST_FORMAT = 'test_format'
TARGET_NAME_TEST_FORMAT_PYTHON = TARGET_NAME_TEST_FORMAT + '_python'
TARGET_NAME_TEST_FORMAT_CPP = TARGET_NAME_TEST_FORMAT + '_cpp'
TARGET_NAME_TEST_FORMAT_MD = TARGET_NAME_TEST_FORMAT + '_md'
TARGET_NAME_TEST_FORMAT_YAML = TARGET_NAME_TEST_FORMAT + '_yaml'
TARGET_NAME_FORMAT = 'format'
TARGET_NAME_FORMAT_PYTHON = TARGET_NAME_FORMAT + '_python'
TARGET_NAME_FORMAT_CPP = TARGET_NAME_FORMAT + '_cpp'
TARGET_NAME_FORMAT_MD = TARGET_NAME_FORMAT + '_md'
TARGET_NAME_FORMAT_YAML = TARGET_NAME_FORMAT + '_yaml'
TARGET_NAME_DEPENDENCIES_CHECK = 'check_dependencies'
TARGET_NAME_GITHUB_ACTIONS_CHECK = 'check_github_actions'
TARGET_NAME_GITHUB_ACTIONS_UPDATE = 'update_github_actions'
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
TARGET_NAME_TESTS_CPP = TARGET_NAME_TESTS + '_cpp'
TARGET_NAME_TESTS_PYTHON = TARGET_NAME_TESTS + '_python'
TARGET_NAME_APIDOC = 'apidoc'
TARGET_NAME_APIDOC_CPP = TARGET_NAME_APIDOC + '_cpp'
TARGET_NAME_APIDOC_PYTHON = TARGET_NAME_APIDOC + '_python'
TARGET_NAME_DOC = 'doc'

VALID_TARGETS = {
    TARGET_NAME_INCREMENT_DEVELOPMENT_VERSION, TARGET_NAME_RESET_DEVELOPMENT_VERSION,
    TARGET_NAME_APPLY_DEVELOPMENT_VERSION, TARGET_NAME_INCREMENT_PATCH_VERSION, TARGET_NAME_INCREMENT_MINOR_VERSION,
    TARGET_NAME_INCREMENT_MAJOR_VERSION, TARGET_NAME_VALIDATE_CHANGELOG_BUGFIX, TARGET_NAME_VALIDATE_CHANGELOG_FEATURE,
    TARGET_NAME_VALIDATE_CHANGELOG_MAIN, TARGET_NAME_UPDATE_CHANGELOG_BUGFIX, TARGET_NAME_UPDATE_CHANGELOG_FEATURE,
    TARGET_NAME_UPDATE_CHANGELOG_MAIN, TARGET_NAME_PRINT_VERSION, TARGET_NAME_PRINT_LATEST_CHANGELOG,
    TARGET_NAME_TEST_FORMAT, TARGET_NAME_TEST_FORMAT_PYTHON, TARGET_NAME_TEST_FORMAT_CPP, TARGET_NAME_TEST_FORMAT_MD,
    TARGET_NAME_TEST_FORMAT_YAML, TARGET_NAME_FORMAT, TARGET_NAME_FORMAT_PYTHON, TARGET_NAME_FORMAT_CPP,
    TARGET_NAME_FORMAT_MD, TARGET_NAME_FORMAT_YAML, TARGET_NAME_DEPENDENCIES_CHECK, TARGET_NAME_GITHUB_ACTIONS_CHECK,
    TARGET_NAME_GITHUB_ACTIONS_UPDATE, TARGET_NAME_VENV, TARGET_NAME_COMPILE, TARGET_NAME_COMPILE_CPP,
    TARGET_NAME_COMPILE_CYTHON, TARGET_NAME_INSTALL, TARGET_NAME_INSTALL_CPP, TARGET_NAME_INSTALL_CYTHON,
    TARGET_NAME_BUILD_WHEELS, TARGET_NAME_INSTALL_WHEELS, TARGET_NAME_TESTS, TARGET_NAME_TESTS_CPP,
    TARGET_NAME_TESTS_PYTHON, TARGET_NAME_APIDOC, TARGET_NAME_APIDOC_CPP, TARGET_NAME_APIDOC_PYTHON, TARGET_NAME_DOC
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

# Defines targets for updating the project's version...
__create_phony_target(env, TARGET_NAME_INCREMENT_DEVELOPMENT_VERSION, action=increment_development_version)
__create_phony_target(env, TARGET_NAME_RESET_DEVELOPMENT_VERSION, action=reset_development_version)
__create_phony_target(env, TARGET_NAME_APPLY_DEVELOPMENT_VERSION, action=apply_development_version)
__create_phony_target(env, TARGET_NAME_INCREMENT_PATCH_VERSION, action=increment_patch_version)
__create_phony_target(env, TARGET_NAME_INCREMENT_MINOR_VERSION, action=increment_minor_version)
__create_phony_target(env, TARGET_NAME_INCREMENT_MAJOR_VERSION, action=increment_major_version)

# Define targets for validating changelogs...
__create_phony_target(env, TARGET_NAME_VALIDATE_CHANGELOG_BUGFIX, action=validate_changelog_bugfix)
__create_phony_target(env, TARGET_NAME_VALIDATE_CHANGELOG_FEATURE, action=validate_changelog_feature)
__create_phony_target(env, TARGET_NAME_VALIDATE_CHANGELOG_MAIN, action=validate_changelog_main)

# Define targets for updating the project's changelog...
__create_phony_target(env, TARGET_NAME_UPDATE_CHANGELOG_BUGFIX, action=update_changelog_bugfix)
__create_phony_target(env, TARGET_NAME_UPDATE_CHANGELOG_FEATURE, action=update_changelog_feature)
__create_phony_target(env, TARGET_NAME_UPDATE_CHANGELOG_MAIN, action=update_changelog_main)

# Define targets for printing information about the project...
__create_phony_target(env, TARGET_NAME_PRINT_VERSION, action=print_current_version)
__create_phony_target(env, TARGET_NAME_PRINT_LATEST_CHANGELOG, action=print_latest_changelog)

# Define targets for checking code style definitions...
target_test_format_python = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_PYTHON, action=check_python_code_style)
target_test_format_cpp = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_CPP, action=check_cpp_code_style)
target_test_format_md = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_MD, action=check_md_code_style)
target_test_format_yaml = __create_phony_target(env, TARGET_NAME_TEST_FORMAT_YAML, action=check_yaml_code_style)
target_test_format = __create_phony_target(env, TARGET_NAME_TEST_FORMAT)
env.Depends(target_test_format,
            [target_test_format_python, target_test_format_cpp, target_test_format_md, target_test_format_yaml])

# Define targets for enforcing code style definitions...
target_format_python = __create_phony_target(env, TARGET_NAME_FORMAT_PYTHON, action=enforce_python_code_style)
target_format_cpp = __create_phony_target(env, TARGET_NAME_FORMAT_CPP, action=enforce_cpp_code_style)
target_format_md = __create_phony_target(env, TARGET_NAME_FORMAT_MD, action=enforce_md_code_style)
target_format_yaml = __create_phony_target(env, TARGET_NAME_FORMAT_YAML, action=enforce_yaml_code_style)
target_format = __create_phony_target(env, TARGET_NAME_FORMAT)
env.Depends(target_format, [target_format_python, target_format_cpp, target_format_md, target_format_yaml])

# Define target for checking dependency versions...
__create_phony_target(env, TARGET_NAME_DEPENDENCIES_CHECK, action=check_dependency_versions)

# Define target for checking and updating the versions of GitHub Actions...
__create_phony_target(env, TARGET_NAME_GITHUB_ACTIONS_CHECK, action=check_github_actions)
__create_phony_target(env, TARGET_NAME_GITHUB_ACTIONS_UPDATE, action=update_github_actions)

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

    for subproject in PYTHON_MODULE.find_subprojects(return_all=True):
        env.Clean([target_install_cpp, DEFAULT_TARGET], subproject.find_shared_libraries())

if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL_CYTHON in COMMAND_LINE_TARGETS \
        or TARGET_NAME_INSTALL in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing extension modules from source tree...')

    for subproject in PYTHON_MODULE.find_subprojects(return_all=True):
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

    for subproject in PYTHON_MODULE.find_subprojects(return_all=True):
        env.Clean([target_build_wheels, DEFAULT_TARGET], subproject.build_dirs)

# Define targets for running automated tests...
target_tests_cpp = __create_phony_target(env, TARGET_NAME_TESTS_CPP, action=tests_cpp)
env.Depends(target_tests_cpp, target_compile_cpp)

target_tests_python = __create_phony_target(env, TARGET_NAME_TESTS_PYTHON, action=tests_python)
env.Depends(target_tests_python, target_install_wheels)

target_tests = __create_phony_target(env, TARGET_NAME_TESTS)
env.Depends(target_tests, [target_tests_cpp, target_tests_python])

# Define targets for generating the documentation...
commands_apidoc_cpp = []
commands_apidoc_python = []

for subproject in CPP_MODULE.find_subprojects():
    apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
    build_files = apidoc_subproject.find_build_files()
    targets_apidoc_cpp = build_files if build_files else apidoc_subproject.build_dir
    command_apidoc_cpp = env.Command(targets_apidoc_cpp, subproject.find_source_files(), action=apidoc_cpp)
    env.NoClean(command_apidoc_cpp)
    commands_apidoc_cpp.append(command_apidoc_cpp)

command_apidoc_cpp_tocfile = env.Command(DOC_MODULE.apidoc_tocfile_cpp, None, action=apidoc_cpp_tocfile)
env.NoClean(command_apidoc_cpp_tocfile)
env.Depends(command_apidoc_cpp_tocfile, commands_apidoc_cpp)

target_apidoc_cpp = env.Alias(TARGET_NAME_APIDOC_CPP, None, None)
env.Depends(target_apidoc_cpp, command_apidoc_cpp_tocfile)

for subproject in PYTHON_MODULE.find_subprojects():
    apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
    build_files = apidoc_subproject.find_build_files()
    targets_apidoc_python = build_files if build_files else apidoc_subproject.build_dir
    command_apidoc_python = env.Command(targets_apidoc_python, subproject.find_source_files(), action=apidoc_python)
    env.NoClean(command_apidoc_python)
    env.Depends(command_apidoc_python, target_install_wheels)
    commands_apidoc_python.append(command_apidoc_python)

command_apidoc_python_tocfile = env.Command(DOC_MODULE.apidoc_tocfile_python, None, action=apidoc_python_tocfile)
env.NoClean(command_apidoc_python_tocfile)
env.Depends(command_apidoc_python_tocfile, commands_apidoc_python)

target_apidoc_python = env.Alias(TARGET_NAME_APIDOC_PYTHON, None, None)
env.Depends(target_apidoc_python, command_apidoc_python_tocfile)

target_apidoc = env.Alias(TARGET_NAME_APIDOC, None, None)
env.Depends(target_apidoc, [target_apidoc_cpp, target_apidoc_python])

doc_files = DOC_MODULE.find_build_files()
targets_doc = doc_files if doc_files else DOC_MODULE.build_dir
command_doc = env.Command(targets_doc, DOC_MODULE.find_source_files(), action=doc)
env.Depends(command_doc, target_apidoc)
target_doc = env.Alias(TARGET_NAME_DOC, None, None)
env.Depends(target_doc, command_doc)

# Define target for cleaning up the documentation and associated build directories...
if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC_CPP in COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing C++ API documentation...')
    env.Clean([target_apidoc_cpp, DEFAULT_TARGET], DOC_MODULE.apidoc_tocfile_cpp)

    for subproject in CPP_MODULE.find_subprojects(return_all=True):
        apidoc_subproject = DOC_MODULE.get_cpp_apidoc_subproject(subproject)
        env.Clean([target_apidoc_cpp, DEFAULT_TARGET], apidoc_subproject.build_dir)

if not COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC_PYTHON in COMMAND_LINE_TARGETS \
        or TARGET_NAME_APIDOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing Python API documentation...')
    env.Clean([target_apidoc_python, DEFAULT_TARGET], DOC_MODULE.apidoc_tocfile_python)

    for subproject in PYTHON_MODULE.find_subprojects(return_all=True):
        apidoc_subproject = DOC_MODULE.get_python_apidoc_subproject(subproject)
        env.Clean([target_apidoc_python, DEFAULT_TARGET], apidoc_subproject.build_dir)

if not COMMAND_LINE_TARGETS or TARGET_NAME_DOC in COMMAND_LINE_TARGETS:
    __print_if_clean(env, 'Removing documentation...')
    env.Clean([target_doc, DEFAULT_TARGET], DOC_MODULE.build_dir)

# Set the default target...
env.Default(DEFAULT_TARGET)
