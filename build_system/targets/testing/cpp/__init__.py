"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for testing C++ code.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder

from targets.compilation.cpp import COMPILE_CPP
from targets.project import Project
from targets.testing.cpp.modules import CppTestModule
from targets.testing.cpp.targets import TestCpp

TESTS_CPP = 'tests_cpp'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
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
