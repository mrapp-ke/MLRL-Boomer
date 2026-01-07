"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "isort".
"""
from core.build_unit import BuildUnit

from targets.code_style.formatter import CodeFormatterProgram
from targets.code_style.modules import CodeModule


class PyLint(CodeFormatterProgram):
    """
    Allows to run the external program "pylint".
    """

    ENABLED_EXTENSIONS = [
        "useless-suppression",
    ]

    def __init__(self, build_unit: BuildUnit, module: CodeModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit,
                         module,
                         'pylint',
                         '--jobs=0',
                         '--ignore=build',
                         '--rcfile=' + str(build_unit.root_directory / '.pylintrc.toml'),
                         '--score=n',
                         cache_file_name='pylint')

        for extension in self.ENABLED_EXTENSIONS:
            self.add_arguments('--enable=' + extension, '--fail-on=' + extension)
