"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "autoflake".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class Autoflake(Program):
    """
    Allows to run the external program "autoflake".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('autoflake', '--config', path.join(build_unit.root_directory, '.autoflake.toml'), '--jobs',
                         '0', *module.find_source_files())
        self.add_conditional_arguments(enforce_changes, '--in-place', '--verbose')
        self.add_conditional_arguments(not enforce_changes, '--check-diff', '--quiet')
        self.set_build_unit(build_unit)
