"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run automated tests via the external program "meson".
"""
from compilation.meson import Meson
from core.build_unit import BuildUnit
from testing.cpp.modules import CppTestModule


class MesonTest(Meson):
    """
    Allows to run the external program "meson test".
    """

    def __init__(self, build_unit: BuildUnit, module: CppTestModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit, 'test', '-C', module.build_directory, '--verbose')
        self.add_conditional_arguments(module.fail_fast, '--maxfail', '1')
        self.install_program(False)
