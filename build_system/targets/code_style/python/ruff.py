"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "ruff".
"""

from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class RuffProgram(CodeFormatterProgram):
    """
    An abstract base class for all classes that run the external program "ruff".
    """

    def __init__(
        self,
        build_unit: BuildUnit,
        module: CodeModule,
        ruff_command: str,
        *arguments: str,
        cache_file_name: str | None = None,
    ):
        super().__init__(
            build_unit,
            module,
            'ruff',
            ruff_command,
            '--config',
            str(build_unit.root_directory / '.ruff.toml'),
            *arguments,
            cache_file_name=cache_file_name,
        )


class RuffFormat(RuffProgram):
    """
    Allows to run the external program "ruff format".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__(
            build_unit,
            module,
            'format',
            cache_file_name='ruff_format' + ('_enforce_changes' if enforce_changes else ''),
        )
        self.add_conditional_arguments(not enforce_changes, '--check')


class RuffCheck(RuffProgram):
    """
    Allows to run the external program "ruff check".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """ """
        super().__init__(
            build_unit,
            module,
            'check',
            cache_file_name='ruff_check' + ('_enforce_changes' if enforce_changes else ''),
        )
        self.add_conditional_arguments(enforce_changes, '--fix')
