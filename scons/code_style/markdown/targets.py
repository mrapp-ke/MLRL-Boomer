"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and enforcing code style definitions for Markdown files.
"""
from code_style.markdown.mdformat import MdFormat
from code_style.modules import CodeModule
from util.files import FileType
from util.modules import ModuleRegistry
from util.targets import PhonyTarget
from util.units import BuildUnit

MODULE_FILTER = CodeModule.Filter(FileType.markdown())


class CheckMarkdownCodeStyle(PhonyTarget.Runnable):
    """
    Checks if Markdown files adhere to the code style definitions. If this is not the case, an error is raised.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            print('Checking Markdown code style in the directory "' + module.root_directory + '"...')
            MdFormat(build_unit, module).run()


class EnforceMarkdownCodeStyle(PhonyTarget.Runnable):
    """
    Enforces Markdown files to adhere to the code style definitions.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            print('Formatting Markdown files in the directory "' + module.root_directory + '"...')
            MdFormat(build_unit, module, enforce_changes=True).run()
