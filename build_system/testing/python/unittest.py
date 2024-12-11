"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run automated tests via the external program "unittest".
"""
from core.build_unit import BuildUnit
from testing.python.modules import PythonTestModule
from util.run import PythonModule


class UnitTest:
    """
    Allows to run the external program "unittest".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonTestModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        self.build_unit = build_unit
        self.module = module

    def run(self):
        """
        Runs the program.
        """
        for test_directory in self.module.find_test_directories():
            PythonModule('xmlrunner', 'discover', '--verbose', '--start-directory', test_directory, '--output',
                         self.module.test_result_directory) \
                .add_conditional_arguments(self.module.fail_fast, '--failfast') \
                .print_arguments(True) \
                .install_program(False) \
                .add_dependencies('unittest-xml-reporting') \
                .set_build_unit(self.build_unit) \
                .run()
