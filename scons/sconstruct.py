"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines the individual targets of the build process.
"""
import sys

from functools import reduce
from os import path

from code_style import check_cpp_code_style, check_python_code_style, enforce_cpp_code_style, enforce_python_code_style
from compilation import compile_cpp, setup_cpp
from modules import BUILD_MODULE, CPP_MODULE
from run import install_runtime_dependencies
from SCons.Script import COMMAND_LINE_TARGETS
from SCons.Script.SConscript import SConsEnvironment


def __create_phony_target(environment, target, action=None):
    return environment.AlwaysBuild(environment.Alias(target, None, action))


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

VALID_TARGETS = {
    TARGET_NAME_TEST_FORMAT, TARGET_NAME_TEST_FORMAT_PYTHON, TARGET_NAME_TEST_FORMAT_CPP, TARGET_NAME_FORMAT,
    TARGET_NAME_FORMAT_PYTHON, TARGET_NAME_FORMAT_CPP, TARGET_NAME_VENV, TARGET_NAME_COMPILE, TARGET_NAME_COMPILE_CPP
}

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
