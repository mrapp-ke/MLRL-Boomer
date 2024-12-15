"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "doxygen".
"""
from os import environ, path
from typing import Dict

from core.build_unit import BuildUnit
from util.env import set_env
from util.io import create_directories
from util.run import Program

from targets.documentation.cpp.modules import CppApidocModule


class Doxygen(Program):
    """
    Allows to run the external program "doxygen".
    """

    @staticmethod
    def __create_environment(module: CppApidocModule) -> Dict:
        env = environ.copy()
        set_env(env, 'DOXYGEN_PROJECT_NAME', 'libmlrl' + module.project_name)
        set_env(env, 'DOXYGEN_INPUT_DIR', module.include_directory)
        set_env(env, 'DOXYGEN_OUTPUT_DIR', module.output_directory)
        set_env(env, 'DOXYGEN_PREDEFINED', 'MLRL' + module.project_name.upper() + '_API=')
        return env

    def __init__(self, build_unit: BuildUnit, module: CppApidocModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('doxygen', path.join(build_unit.root_directory, 'Doxyfile'))
        self.module = module
        self.print_arguments(True)
        self.install_program(False)
        self.use_environment(self.__create_environment(module))
        self.set_build_unit(build_unit)

    def _before(self):
        create_directories(self.module.output_directory)
