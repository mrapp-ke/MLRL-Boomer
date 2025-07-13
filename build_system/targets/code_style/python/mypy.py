"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "mypy".
"""
from os import path

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
        super().__init__('mypy', '--config-file', path.join(build_unit.root_directory, '.mypy.ini'),
                         '--warn-unused-configs', *module.find_source_files())
        self.set_build_unit(build_unit)
        self.module = module

    def _should_be_skipped(self) -> bool:
        blacklist = {
            'build_system',
            path.join('python', 'subprojects', 'common'),
            path.join('python', 'subprojects', 'util'),
            path.join('python', 'subprojects', 'testbed'),
            path.join('python', 'subprojects', 'boosting'),
        }
        return self.module.root_directory in blacklist
