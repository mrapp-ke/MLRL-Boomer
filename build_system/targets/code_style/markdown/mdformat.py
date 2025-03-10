"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "mdformat".
"""
from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class MdFormat(Program):
    """
    Allows to run the external program "mdformat".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('mdformat', '--number', '--wrap', 'no', '--end-of-line', 'lf')
        self.add_conditional_arguments(not enforce_changes, '--check')
        self.add_arguments(*module.find_source_files())
        self.set_build_unit(build_unit)
        self.add_dependencies('mdformat-myst', 'mdformat-deflist')
