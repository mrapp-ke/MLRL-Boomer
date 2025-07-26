"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cython-lint".
"""
from core.build_unit import BuildUnit
from util.run import Program

from targets.code_style.modules import CodeModule


class CythonLint(Program):
    """
    Allows to run the external program "cython-lint".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('cython-lint', *map(str, module.find_source_files()))
        self.set_build_unit(build_unit)
