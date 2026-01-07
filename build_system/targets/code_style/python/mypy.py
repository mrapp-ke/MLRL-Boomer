"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "mypy".
"""
from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class Mypy(CodeFormatterProgram):
    """
    Allows to run the external program "mypy".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit,
                         module,
                         'mypy',
                         '--config-file',
                         str(build_unit.root_directory / '.mypy.ini'),
                         '--warn-unused-configs',
                         cache_file_name='mypy')
