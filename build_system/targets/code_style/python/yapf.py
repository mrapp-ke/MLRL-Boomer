"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "yapf".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class Yapf(Program):
    """
    Allows to run the external program "yapf".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yapf', '--parallel', '--style=' + path.join(build_unit.root_directory, '.style.yapf'),
                         '--in-place' if enforce_changes else '--diff', *module.find_source_files())
        self.set_build_unit(build_unit)
