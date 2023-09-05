"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines the individual targets of the build process.
"""
from os import path

from code_style import check_cpp_code_style, check_python_code_style, enforce_cpp_code_style, enforce_python_code_style
from modules import BUILD_MODULE
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

# Create temporary file ".sconsign.dblite" in the build directory...
env = SConsEnvironment()
env.SConsignFile(name=path.join(BUILD_MODULE.build_dir, '.sconsign'))

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
