"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for formatting code.
"""
from abc import ABC
from typing import List, Optional, override

from core.build_unit import BuildUnit
from core.changes import ChangeDetection
from util.run import Program

from targets.code_style.modules import CodeModule


class CodeFormatterProgram(Program, ABC):
    """
    An abstract base class for all programs that format code or check the formatting of code.
    """

    def __init__(self,
                 build_unit: BuildUnit,
                 module: CodeModule,
                 program: str,
                 *arguments: str,
                 cache_file_name: Optional[str] = None):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param program:         The name of the program to be run
        :param arguments:       Optional arguments to be passed to the program
        :param cache_file_name: The name of the file that should be used for tracking modified source files or None, if
                                no change detection should be used
        """
        super().__init__(program, *arguments)
        self.set_build_unit(build_unit)
        self.cache_file = BuildUnit().build_directory / (cache_file_name + '.json') if cache_file_name else None
        self.module = module
        self._original_arguments: Optional[List[str]] = None

    @override
    def _should_be_skipped(self) -> bool:
        if not self._original_arguments:
            self._original_arguments = list(self.arguments)

        module = self.module
        source_files = module.find_source_files()
        cache_file = self.cache_file

        if cache_file:
            source_files = ChangeDetection(cache_file).get_changed_files(module, *source_files)

        if source_files:
            self.arguments = self._original_arguments + list(map(str, source_files))
            return False

        return True

    @override
    def _after(self):
        cache_file = self.cache_file

        if cache_file:
            module = self.module
            ChangeDetection(cache_file).track_files(module, *module.find_source_files())
