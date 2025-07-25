"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating API documentations for Python code.
"""
from typing import List, cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.log import Log

from targets.documentation.python.modules import PythonApidocModule
from targets.documentation.python.sphinx_apidoc import SphinxApidoc
from targets.documentation.targets import ApidocIndex

MODULE_FILTER = PythonApidocModule.Filter()


class ApidocPython(BuildTarget.Runnable):
    """
    Generates API documentations for Python code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        apidoc_module = cast(PythonApidocModule, module)
        Log.info('Generating Python API documentation for directory "%s"...', apidoc_module.root_directory)
        SphinxApidoc(build_unit, apidoc_module).run()

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        apidoc_module = cast(PythonApidocModule, module)
        return [apidoc_module.output_directory]

    @override
    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        apidoc_module = cast(PythonApidocModule, module)
        return apidoc_module.find_source_files()

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        apidoc_module = cast(PythonApidocModule, module)
        Log.info('Removing Python API documentation for directory "%s"...', apidoc_module.root_directory)
        return super().get_clean_files(build_unit, apidoc_module)


class ApidocIndexPython(ApidocIndex):
    """
    Generates index files referencing API documentations for Python code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)
