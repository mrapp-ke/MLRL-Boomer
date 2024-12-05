"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for YAML files.
"""
from code_style.modules import CodeModule
from code_style.yaml.yamlfix import YamlFix
from util.files import FileType
from util.modules import ModuleRegistry
from util.targets import PhonyTarget
from util.units import BuildUnit

MODULE_FILTER = CodeModule.Filter(FileType.yaml())


class CheckYamlCodeStyle(PhonyTarget.Runnable):
    """
    Checks if YAML files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            print('Checking YAML files in the directory "' + module.root_directory + '"...')
            YamlFix(build_unit, module).run()


class EnforceYamlCodeStyle(PhonyTarget.Runnable):
    """
    Enforces YAML files to adhere to the code style definitions.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            print('Formatting YAML files in the directory "' + module.root_directory + '"...')
            YamlFix(build_unit, module, enforce_changes=True).run()
