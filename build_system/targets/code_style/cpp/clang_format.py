"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "clang-format".
"""
from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class ClangFormat(CodeFormatterProgram):
    """
    Allows to run the external program "clang-format".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__(build_unit,
                         module,
                         'clang-format',
                         '--style=file:' + str(build_unit.root_directory / '.clang-format'),
                         cache_file_name='clang_format' + ('_enforce_changes' if enforce_changes else ''))
        self.add_conditional_arguments(enforce_changes, '-i')
        self.add_conditional_arguments(not enforce_changes, '--dry-run', '--Werror')
