"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cpplint".
"""
from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class CppLint(CodeFormatterProgram):
    """
    Allows to run the external program "cpplint".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit, module, 'cpplint', '--quiet', '--config=.cpplint.cfg', cache_file_name='cpplint')
