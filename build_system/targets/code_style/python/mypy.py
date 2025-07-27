"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "mypy".
"""
from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class Mypy(Program):
    """
    Allows to run the external program "mypy".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('mypy', '--config-file', str(build_unit.root_directory / '.mypy.ini'), '--warn-unused-configs',
                         *map(str, module.find_source_files()))
        self.set_build_unit(build_unit)
        self.module = module
