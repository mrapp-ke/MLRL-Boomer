"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for C++ files.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.files import FileType
from util.log import Log

from targets.code_style.cpp.clang_format import ClangFormat
from targets.code_style.cpp.cpplint import CppLint
from targets.code_style.modules import CodeModule

MODULE_FILTER = CodeModule.Filter(FileType.cpp())


class CheckCppCodeStyle(PhonyTarget.Runnable):
    """
    Checks if C++ source files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Checking C++ code style in directory "%s"...', module.root_directory)
        ClangFormat(build_unit, module).run()
        CppLint(build_unit, module).run()


class EnforceCppCodeStyle(PhonyTarget.Runnable):
    """
    Enforces C++ source files to adhere to the code style definitions.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Formatting C++ code in directory "%s"...', module.root_directory)
        ClangFormat(build_unit, module, enforce_changes=True).run()
