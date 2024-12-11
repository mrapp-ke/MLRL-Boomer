"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "isort".
"""
from code_style.modules import CodeModule
from core.build_unit import BuildUnit
from util.run import Program


class ISort(Program):
    """
    Allows to run the external program "isort".
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('isort', '--settings-path', build_unit.root_directory, '--virtual-env', 'venv',
                         '--skip-gitignore', *module.find_source_files())
        self.add_conditional_arguments(not enforce_changes, '--check')
        self.set_build_unit(build_unit)
