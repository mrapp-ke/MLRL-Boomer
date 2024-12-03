"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for testing C++ code.
"""
from compilation.cpp import COMPILE_CPP
from testing.cpp.targets import TestCpp
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

TESTS_CPP = 'tests_cpp'

TARGETS = TargetBuilder(BuildUnit('testing', 'cpp')) \
    .add_phony_target(TESTS_CPP) \
        .depends_on(COMPILE_CPP) \
        .set_runnables(TestCpp()) \
    .build()
