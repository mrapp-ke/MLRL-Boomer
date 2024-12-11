"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "breathe-apidoc".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import Program

from targets.documentation.cpp.modules import CppApidocModule


class BreatheApidoc(Program):
    """
    Allows to run the external program "breathe-apidoc".
    """

    def __init__(self, build_unit: BuildUnit, module: CppApidocModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('breathe-apidoc', '--members', '--project', module.project_name, '-g', 'file', '-o',
                         module.output_directory, path.join(module.output_directory, 'xml'))
        self.module = module
        self.print_arguments(True)
        self.install_program(False)
        self.add_dependencies('breathe')
        self.set_build_unit(build_unit)
