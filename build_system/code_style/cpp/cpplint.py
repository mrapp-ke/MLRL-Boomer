"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cpplint".
"""
from code_style.modules import CodeModule
from util.run import Program
from util.units import BuildUnit


class CppLint(Program):
    """
    Allows to run the external program "cpplint".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('cpplint', '--quiet', '--config=.cpplint.cfg', *module.find_source_files())
        self.set_build_unit(build_unit)
