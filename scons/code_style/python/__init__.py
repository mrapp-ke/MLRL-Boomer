"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for checking and enforcing code style definitions for Python and Cython files.
"""
from code_style.python.targets import CheckCythonCodeStyle, CheckPythonCodeStyle, EnforceCythonCodeStyle, \
    EnforcePythonCodeStyle
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_PYTHON = 'format_python'

TEST_FORMAT_PYTHON = 'test_format_python'

TARGETS = TargetBuilder(BuildUnit('code_style', 'python')) \
    .add_phony_target(FORMAT_PYTHON).set_runnables(EnforcePythonCodeStyle(), EnforceCythonCodeStyle()) \
    .add_phony_target(TEST_FORMAT_PYTHON).set_runnables(CheckPythonCodeStyle(), CheckCythonCodeStyle()) \
    .build()
