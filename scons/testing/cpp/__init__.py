"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for testing C++ code.
"""
from compilation.cpp import COMPILE_CPP
from testing.cpp.targets import TestCpp
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

TARGETS = TargetBuilder(BuildUnit('testing', 'cpp')) \
    .add_phony_target('tests_cpp') \
        .depends_on(COMPILE_CPP) \
        .set_runnables(TestCpp()) \
    .build()
