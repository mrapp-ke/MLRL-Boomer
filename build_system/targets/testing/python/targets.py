"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for testing Python code.
"""
from typing import cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.log import Log

from targets.testing.python.modules import PythonTestModule
from targets.testing.python.pytest import Pytest


class TestPython(PhonyTarget.Runnable):
    """
    Runs automated tests for Python code.
    """

    def __init__(self):
        super().__init__(PythonTestModule.Filter())

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        test_module = cast(PythonTestModule, module)
        Log.info('Running tests in directory "%s"...', test_module.root_directory)
        Pytest(build_unit, test_module).run()
