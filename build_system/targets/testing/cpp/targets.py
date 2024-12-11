"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing C++ code.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget, PhonyTarget

from targets.testing.cpp.meson import MesonTest
from targets.testing.cpp.modules import CppTestModule


class TestCpp(PhonyTarget.Runnable):
    """
    Runs automated tests for C++ code.
    """

    def __init__(self):
        super().__init__(CppTestModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        MesonTest(build_unit, module).run()
