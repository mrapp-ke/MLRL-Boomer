"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "yapf".
"""
from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class Yapf(CodeFormatterProgram):
    """
    Allows to run the external program "yapf".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__(build_unit,
                         module,
                         'yapf',
                         '--parallel',
                         '--style=' + str(build_unit.root_directory / '.style.yapf'),
                         '--in-place' if enforce_changes else '--diff',
                         cache_file_name='yapf' + ('_enforce_changes' if enforce_changes else ''))
