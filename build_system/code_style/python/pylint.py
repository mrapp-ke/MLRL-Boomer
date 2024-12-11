"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "isort".
"""
from os import path

from code_style.modules import CodeModule
from core.build_unit import BuildUnit
from util.run import Program


class PyLint(Program):
    """
    Allows to run the external program "pylint".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('pylint', *module.find_source_files(), '--jobs=0', '--ignore=build',
                         '--rcfile=' + path.join(build_unit.root_directory, '.pylintrc'), '--score=n')
        self.set_build_unit(build_unit)
