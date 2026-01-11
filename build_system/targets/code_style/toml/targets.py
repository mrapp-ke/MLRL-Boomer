"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for TOML files.
"""
from typing import cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.files import FileType
from util.log import Log

from targets.code_style.modules import CodeModule
from targets.code_style.toml.taplo import TaploFormat, TaploLint

MODULE_FILTER = CodeModule.Filter(FileType.toml())


class CheckTomlCodeStyle(PhonyTarget.Runnable):
    """
    Checks if TOML files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        code_module = cast(CodeModule, module)
        Log.info('Checking TOML files in the directory "%s"...', code_module.root_directory)
        TaploLint(build_unit, code_module).run()
        TaploFormat(build_unit, code_module).run()


class EnforceTomlCodeStyle(PhonyTarget.Runnable):
    """
    Enforces TOML files to adhere to the code style definitions.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        code_module = cast(CodeModule, module)
        Log.info('Formatting TOML files in the directory "%s"...', code_module.root_directory)
        TaploFormat(build_unit, code_module, enforce_changes=True).run()
