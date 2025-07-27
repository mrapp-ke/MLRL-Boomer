"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating API documentations for C++ code.
"""
from pathlib import Path
from typing import List, cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget, PhonyTarget
from util.log import Log

from targets.documentation.cpp.breathe_apidoc import BreatheApidoc
from targets.documentation.cpp.doxygen import Doxygen, DoxygenUpdate
from targets.documentation.cpp.modules import CppApidocModule
from targets.documentation.targets import ApidocIndex

MODULE_FILTER = CppApidocModule.Filter()


class ApidocCpp(BuildTarget.Runnable):
    """
    Generates API documentations for C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        apidoc_module = cast(CppApidocModule, module)
        Log.info('Generating C++ API documentation for directory "%s"...', apidoc_module.root_directory)
        Doxygen(build_unit, apidoc_module).run()
        BreatheApidoc(build_unit, apidoc_module).run()

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[Path]:
        apidoc_module = cast(CppApidocModule, module)
        return [apidoc_module.output_directory]

    @override
    def get_input_files(self, _: BuildUnit, module: Module) -> List[Path]:
        apidoc_module = cast(CppApidocModule, module)
        return apidoc_module.find_header_files()

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[Path]:
        apidoc_module = cast(CppApidocModule, module)
        Log.info('Removing C++ API documentation for directory "%s"...', apidoc_module.root_directory)
        return super().get_clean_files(build_unit, module)


class ApidocIndexCpp(ApidocIndex):
    """
    Generates index files referencing API documentations for C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)


class UpdateDoxyfile(PhonyTarget.Runnable):
    """
    Updates the Doxyfile used for generating API documentations for C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run_all(self, build_unit: BuildUnit, _: List[Module]):
        Log.info('Updating Doxyfile in directory "%s"...', build_unit.root_directory)
        DoxygenUpdate(build_unit).run()
