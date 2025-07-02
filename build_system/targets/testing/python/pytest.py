"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run automated tests via the external program "pytest".
"""
from functools import reduce
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
                         '--config-file=' + path.join(build_unit.root_directory, '.pytest.ini'), '--strict-config',
                         '--strict-markers', '--junit-xml=' + path.join(module.result_directory, 'junit.xml'),
                         module.root_directory)
        self.add_conditional_arguments(module.fail_fast, '--exitfirst')
        self.add_conditional_arguments(module.only_failed, '--last-failed')
        self.add_conditional_arguments(
            module.markers is not None, '-m',
            reduce(lambda aggr, marker: aggr + (' and ' if aggr else '') + marker, module.markers, ''))
        self.set_accepted_exit_codes(0, 5)

        self.print_arguments(True)
        self.set_build_unit(build_unit)
