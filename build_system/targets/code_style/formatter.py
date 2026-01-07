"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for formatting code.
"""
from abc import ABC
from pathlib import Path
from typing import List, Optional, override

from core.build_unit import BuildUnit
from core.changes import ChangeDetection
from util.run import Program

from targets.code_style.modules import CodeModule


class CodeChangeDetection:
    """
    Allows to keep track of modified source files.
    """

    def __init__(self, module: CodeModule, cache_file_name: str):
        """
        :param module:          A module that provides access to source code
        :param cache_file_name: The name of the file that should be used for tracking modified source files
        """
        self.change_detection = ChangeDetection(BuildUnit().build_directory / (cache_file_name + '.json'))
        self.module = module

    def find_modified_source_files(self) -> List[Path]:
        """
        Finds and returns all modified source files.

        :return: A list that contains the paths of the source files that have been found
        """
        module = self.module
        return self.change_detection.get_changed_files(module, *module.find_source_files())

    def update_cache(self):
        """
        Updates the cache to keep track of source files.
        """
        module = self.module
        self.change_detection.track_files(module, *module.find_source_files())


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
        self.change_detection = CodeChangeDetection(module, cache_file_name) if cache_file_name else None
        self.module = module
        self._original_arguments: Optional[List[str]] = None

    @override
    def _should_be_skipped(self) -> bool:
        if not self._original_arguments:
            self._original_arguments = list(self.arguments)

        module = self.module
        change_detection = self.change_detection
        source_files = change_detection.find_modified_source_files() if change_detection else module.find_source_files()

        if source_files:
            self.arguments = self._original_arguments + list(map(str, source_files))
            return False

        return True

    @override
    def _after(self):
        change_detection = self.change_detection

        if change_detection:
            change_detection.update_cache()
