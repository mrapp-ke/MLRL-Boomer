"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating API documentations for Python code.
"""
from typing import List

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.log import Log

from targets.documentation.python.modules import PythonApidocModule
from targets.documentation.python.sphinx_apidoc import SphinxApidoc


class ApidocPython(BuildTarget.Runnable):
    """
    Generates API documentations for Python code.
    """

    def __init__(self):
        super().__init__(PythonApidocModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Generating Python API documentation for directory "%s"...', module.root_directory)
        SphinxApidoc(build_unit, module).run()

    def get_output_files(self, module: Module) -> List[str]:
        return [module.output_directory]

    def get_input_files(self, module: Module) -> List[str]:
        return module.find_source_files()

    def get_clean_files(self, module: Module) -> List[str]:
        Log.info('Removing Python API documentation for directory "%s"...', module.root_directory)
        return super().get_clean_files(module)
