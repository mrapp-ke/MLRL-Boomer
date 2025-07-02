"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run automated tests via the external program "pytest".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import PythonModule

from targets.testing.python.modules import PythonTestModule


class Pytest(PythonModule):
    """
    Allows to run the external program "pytest".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonTestModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('pytest', '--verbose', '--color=yes',
                         '--junit-xml=' + path.join(module.result_directory, 'junit.xml'), module.root_directory)
        self.add_conditional_arguments(module.fail_fast, '--exitfirst')
        self.print_arguments(True)
        self.set_build_unit(build_unit)
