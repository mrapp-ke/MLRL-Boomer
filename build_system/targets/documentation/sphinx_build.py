"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "sphinx-build".
"""
from os import path

from core.build_unit import BuildUnit
from util.run import Program

from targets.documentation.cpp.modules import CppApidocModule


class SphinxBuild(Program):
    """
    Allows to run the external program "sphinx-build".
    """

    def __init__(self, build_unit: BuildUnit, module: CppApidocModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('sphinx-build', '--jobs', 'auto', module.root_directory,
                         path.join(module.output_directory, 'html'))
        self.module = module
        self.print_arguments(True)
        self.install_program(False)
        self.add_dependencies(
            'furo',
            'myst-parser',
            'sphinx',
            'sphinx-copybutton',
            'sphinx-favicon',
            'sphinx-inline-tabs',
            'sphinxext-opengraph',
        )
        self.set_build_unit(build_unit)
