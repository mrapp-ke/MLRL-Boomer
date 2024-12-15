"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "clang-format".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class ClangFormat(Program):
    """
    Allows to run the external program "clang-format".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('clang-format', '--style=file:' + path.join(build_unit.root_directory, '.clang-format'))
        self.add_conditional_arguments(enforce_changes, '-i')
        self.add_conditional_arguments(not enforce_changes, '--dry-run', '--Werror')
        self.add_arguments(*module.find_source_files())
        self.set_build_unit(build_unit)
