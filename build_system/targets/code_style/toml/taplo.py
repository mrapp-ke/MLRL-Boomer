"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "taplo".
"""
from abc import ABC
from os import environ, path
from typing import Dict, List, Optional

from core.build_unit import BuildUnit
from util.run import Program


class Taplo(Program, ABC):
    """
    An abstract base class for all classes that allow to run the external program "taplo".
    """

    @staticmethod
    def __create_environment() -> Dict:
        env = environ.copy()
        env['RUST_LOG'] = 'warn'
        return env

    def __init__(self, build_unit: BuildUnit, taplo_command: str, *arguments: str):
        """
        :param build_unit:  The build unit from which the program should be run
        :param program:     The taplo command to be run
        :param arguments:   Optional arguments to be passed to taplo
        """
        super().__init__('taplo', taplo_command, *arguments)
        self.set_build_unit(build_unit)
        self.print_arguments(True)
        self.use_environment(self.__create_environment())


class TaploFormat(Taplo):
    """
    Allows to run the external program "taplo format".
    """

    def __init__(self, build_unit: BuildUnit, files: List[str], enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param files:           A list that contains the files, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__(build_unit, 'format', '--config', path.join(build_unit.root_directory, '.taplo.toml'))
        self.add_conditional_arguments(not enforce_changes, '--check')
        self.add_arguments(*files)


class TaploLint(Taplo):
    """
    Allows to run the external program "taplo lint".
    """

    def __init__(self, build_unit: BuildUnit, files: List[str], schema: Optional[str] = None):
        """
        :param build_unit:      The build unit from which the program should be run
        :param files:           A list that contains the files, the program should be applied to
        :param schema:          An optional URL to the schema that should be used for validation
        """
        super().__init__(build_unit, 'lint')
        self.add_conditional_arguments(schema is not None, '--schema', schema)
        self.add_arguments(*files)
