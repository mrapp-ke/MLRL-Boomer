"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for Python and Cython files.
"""
from code_style.modules import CodeModule
from code_style.python.isort import ISort
from code_style.python.pylint import PyLint
from code_style.python.yapf import Yapf
from util.files import FileType
from util.log import Log
from util.modules import ModuleRegistry
from util.targets import PhonyTarget
from util.units import BuildUnit

PYTHON_MODULE_FILTER = CodeModule.Filter(FileType.python())

CYTHON_MODULE_FILTER = CodeModule.Filter(FileType.cython())


class CheckPythonCodeStyle(PhonyTarget.Runnable):
    """
    Checks if Python source files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(PYTHON_MODULE_FILTER):
            Log.info('Checking Python code style in directory "%s"...', module.root_directory)
            ISort(build_unit, module).run()
            Yapf(build_unit, module).run()
            PyLint(build_unit, module).run()


class EnforcePythonCodeStyle(PhonyTarget.Runnable):
    """
    Enforces Python source files to adhere to the code style definitions.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(PYTHON_MODULE_FILTER):
            Log.info('Formatting Python code in directory "%s"...', module.root_directory)
            ISort(build_unit, module, enforce_changes=True).run()
            Yapf(build_unit, module, enforce_changes=True).run()


class CheckCythonCodeStyle(PhonyTarget.Runnable):
    """
    Checks if Cython source files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(CYTHON_MODULE_FILTER):
            Log.info('Checking Cython code style in directory "%s"...', module.root_directory)
            ISort(build_unit, module).run()


class EnforceCythonCodeStyle(PhonyTarget.Runnable):
    """
    Enforces Cython source files to adhere to the code style definitions.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(CYTHON_MODULE_FILTER):
            Log.info('Formatting Cython code in directory "%s"...', module.root_directory)
            ISort(build_unit, module, enforce_changes=True).run()
