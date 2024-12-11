"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing C++ code.
"""
from testing.cpp.meson import MesonTest
from testing.cpp.modules import CppTestModule
from util.modules import Module
from util.targets import BuildTarget, PhonyTarget
from util.units import BuildUnit


class TestCpp(PhonyTarget.Runnable):
    """
    Runs automated tests for C++ code.
    """

    def __init__(self):
        super().__init__(CppTestModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        MesonTest(build_unit, module).run()
