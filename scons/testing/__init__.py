"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for testing code.
"""
from testing.cpp import TESTS_CPP
from testing.python import TESTS_PYTHON
from util.targets import TargetBuilder
from util.units import BuildUnit

TARGETS = TargetBuilder(BuildUnit('testing')) \
    .add_phony_target('tests') \
        .depends_on(TESTS_CPP, TESTS_PYTHON) \
        .nop() \
    .build()
