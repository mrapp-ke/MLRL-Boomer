"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing Python code.
"""
from testing.python.modules import PythonTestModule
from testing.python.unittest import UnitTest
from util.modules import Module
from util.targets import PhonyTarget
from util.units import BuildUnit


class TestPython(PhonyTarget.Runnable):
    """
    Runs automated tests for Python code.
    """

    def __init__(self):
        super().__init__(PythonTestModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        UnitTest(build_unit, module).run()
