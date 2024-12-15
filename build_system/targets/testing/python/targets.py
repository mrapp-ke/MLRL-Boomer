"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing Python code.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget

from targets.testing.python.modules import PythonTestModule
from targets.testing.python.unittest import UnitTest


class TestPython(PhonyTarget.Runnable):
    """
    Runs automated tests for Python code.
    """

    def __init__(self):
        super().__init__(PythonTestModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        UnitTest(build_unit, module).run()
