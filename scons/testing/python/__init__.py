"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for testing Python code.
"""
from testing.python.targets import TestPython
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

TESTS_PYTHON = 'tests_python'

# TODO .depends_on(INSTALL_WHEELS)
TARGETS = TargetBuilder(BuildUnit('testing', 'python')) \
    .add_phony_target(TESTS_PYTHON) \
        .set_runnables(TestPython()) \
    .build()
