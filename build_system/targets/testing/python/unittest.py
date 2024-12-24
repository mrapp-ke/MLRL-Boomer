"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run automated tests via the external program "unittest".
"""
from core.build_unit import BuildUnit
from util.run import PythonModule

from targets.paths import Project
from targets.testing.python.modules import PythonTestModule


class UnitTest(PythonModule):
    """
    Allows to run the external program "unittest".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonTestModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('xmlrunner', 'discover', '--verbose', '--start-directory', module.root_directory,
                         '--top-level-directory', Project.Python.root_directory, '--output', module.result_directory)
        self.add_conditional_arguments(module.fail_fast, '--failfast')
        self.print_arguments(True)
        self.install_program(False)
        self.add_dependencies('unittest-xml-reporting')
        self.set_build_unit(build_unit)
