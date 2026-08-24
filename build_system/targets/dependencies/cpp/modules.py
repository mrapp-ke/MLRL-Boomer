"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Meson wrap files.
"""

from pathlib import Path
from typing import override

from core.build_unit import BuildUnit
from core.modules import Module, ModuleRegistry
from targets.dependencies.cpp.wrap_file import WrapFile
from util.files import FileSearch


class WrapFileModule(Module):
    """
    A module that provides access to Meson wrap files.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `WrapFileModule`.
        """

        @override
        def matches(self, module: Module, _: ModuleRegistry) -> bool:
            return isinstance(module, WrapFileModule)

    def __init__(self, root_directory: Path, wrap_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:      The path to the module's root directory
        :param wrap_file_search:    The `FileSearch` that should be used to search for Meson wrap files
        """
        self.root_directory = root_directory
        self.wrap_file_search = wrap_file_search

    def find_wrap_files(self, build_unit: BuildUnit) -> list[WrapFile]:
        """
        Finds and returns all Meson wrap files that belong to the module.

        :param build_unit:  The `BuildUnit` from which this method is invoked
        :return:            A list that contains the requirements files that have been found
        """
        return [
            WrapFile(wrap_file)
            for wrap_file in self.wrap_file_search.filter_by_suffix('wrap').list(self.root_directory)
        ]

    @override
    def __str__(self) -> str:
        return f'WrapFileModule {{root_directory="{self.root_directory}"}}'
