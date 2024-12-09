"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing Python code.
"""
from testing.python.modules import PythonTestModule
from testing.python.unittest import UnitTest
from util.modules import ModuleRegistry
from util.targets import PhonyTarget
from util.units import BuildUnit


class TestPython(PhonyTarget.Runnable):
    """
    Runs automated tests for Python code.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(PythonTestModule.Filter()):
            UnitTest(build_unit, module).run()
