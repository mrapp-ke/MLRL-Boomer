"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for testing C++ code.
"""
from compilation.cpp import COMPILE_CPP
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from testing.cpp.modules import CppTestModule
from testing.cpp.targets import TestCpp
from util.paths import Project

TESTS_CPP = 'tests_cpp'

TARGETS = TargetBuilder(BuildUnit('testing', 'cpp')) \
    .add_phony_target(TESTS_CPP) \
        .depends_on(COMPILE_CPP) \
        .set_runnables(TestCpp()) \
    .build()

MODULES = [
    CppTestModule(
        root_directory=Project.Cpp.root_directory,
        build_directory_name=Project.Cpp.build_directory_name,
    ),
]
