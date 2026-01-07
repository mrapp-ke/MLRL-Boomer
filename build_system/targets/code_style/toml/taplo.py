"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "taplo".
"""
from abc import ABC
from os import environ
from typing import Optional

from core.build_unit import BuildUnit
from util.env import Env

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class Taplo(CodeFormatterProgram, ABC):
    """
    An abstract base class for all classes that allow to run the external program "taplo".
    """

    @staticmethod
    def __create_environment() -> Env:
        env = environ.copy()
        env['RUST_LOG'] = 'warn'
        return env

    def __init__(self,
                 build_unit: BuildUnit,
                 module: CodeModule,
                 taplo_command: str,
                 *arguments: str,
                 cache_file_name: Optional[str] = None):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param taplo_command:   The taplo command to be run
        :param arguments:       Optional arguments to be passed to taplo
        :param cache_file_name: The name of the file that should be used for tracking modified source files or None, if
                                no change detection should be used
        """
        super().__init__(build_unit, module, 'taplo', taplo_command, *arguments)
        self.print_arguments(True)
        self.use_environment(self.__create_environment())


class TaploFormat(Taplo):
    """
    Allows to run the external program "taplo format".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__(build_unit,
                         module,
                         'format',
                         '--config',
                         str(build_unit.root_directory / '.taplo.toml'),
                         cache_file_name='taplo_format' + ('_enforce_changes' if enforce_changes else ''))
        self.add_conditional_arguments(not enforce_changes, '--check')


class TaploLint(Taplo):
    """
    Allows to run the external program "taplo lint".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit, module, 'lint', cache_file_name='taplo_lint')
