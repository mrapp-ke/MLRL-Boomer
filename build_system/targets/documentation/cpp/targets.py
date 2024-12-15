"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating API documentations for C++ code.
"""
from typing import List

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.log import Log

from targets.documentation.cpp.breathe_apidoc import BreatheApidoc
from targets.documentation.cpp.doxygen import Doxygen
from targets.documentation.cpp.modules import CppApidocModule
from targets.documentation.targets import ApidocIndex

MODULE_FILTER = CppApidocModule.Filter()


class ApidocCpp(BuildTarget.Runnable):
    """
    Generates API documentations for C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Generating C++ API documentation for directory "%s"...', module.root_directory)
        Doxygen(build_unit, module).run()
        BreatheApidoc(build_unit, module).run()

    def get_output_files(self, module: Module) -> List[str]:
        return [module.output_directory]

    def get_input_files(self, module: Module) -> List[str]:
        return module.find_header_files()

    def get_clean_files(self, module: Module) -> List[str]:
        Log.info('Removing C++ API documentation for directory "%s"...', module.root_directory)
        return super().get_clean_files(module)


class ApidocIndexCpp(ApidocIndex):
    """
    Generates index files referencing API documentations for C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)
