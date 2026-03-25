"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for Cython files.
"""

from typing import cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from targets.code_style.modules import CodeModule
from targets.code_style.cython.autoflake import Autoflake
from targets.code_style.cython.cython_lint import CythonLint
from targets.code_style.cython.isort import ISort
from util.files import FileType
from util.log import Log

CYTHON_MODULE_FILTER = CodeModule.Filter(FileType.cython())


class CheckCythonCodeStyle(PhonyTarget.Runnable):
    """
    Checks if Cython source files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def __init__(self):
        super().__init__(CYTHON_MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        code_module = cast(CodeModule, module)
        Log.info(f'Checking Cython code style in directory "{code_module.root_directory}"...')
        ISort(build_unit, code_module).run()
        CythonLint(build_unit, code_module).run()


class EnforceCythonCodeStyle(PhonyTarget.Runnable):
    """
    Enforces Cython source files to adhere to the code style definitions.
    """

    def __init__(self):
        super().__init__(CYTHON_MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        code_module = cast(CodeModule, module)
        Log.info(f'Formatting Cython code in directory "{code_module.root_directory}"...')
        Autoflake(build_unit, code_module, enforce_changes=True).run()
        ISort(build_unit, code_module, enforce_changes=True).run()
