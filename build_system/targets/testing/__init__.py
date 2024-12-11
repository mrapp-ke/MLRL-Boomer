"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for testing code.
"""
from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.testing.cpp import TESTS_CPP
from targets.testing.python import TESTS_PYTHON

TARGETS = TargetBuilder(BuildUnit('targets', 'testing')) \
    .add_phony_target('tests') \
        .depends_on(TESTS_CPP, TESTS_PYTHON) \
        .nop() \
    .build()
