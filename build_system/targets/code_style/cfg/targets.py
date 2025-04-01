"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for .cfg files.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.files import FileType
from util.log import Log

from targets.code_style.cfg.cfg_formatter import CfgFormatter
from targets.code_style.modules import CodeModule

MODULE_FILTER = CodeModule.Filter(FileType.cfg())


class CheckCfgCodeStyle(PhonyTarget.Runnable):
    """
    Checks if .cfg files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Checking .cfg code style in the directory "%s"...', module.root_directory)
        CfgFormatter(build_unit, module).run()


class EnforceCfgCodeStyle(PhonyTarget.Runnable):
    """
    Enforces .cfg files to adhere to the code style definitions.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Formatting .cfg files in the directory "%s"...', module.root_directory)
        CfgFormatter(build_unit, module, enforce_changes=True).run()
