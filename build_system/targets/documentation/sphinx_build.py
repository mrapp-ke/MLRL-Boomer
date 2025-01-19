"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "sphinx-build".
"""
from os import path

from core.build_unit import BuildUnit
from util.io import delete_files
from util.log import Log
from util.run import Program

from targets.documentation.modules import SphinxModule


class SphinxBuild(Program):
    """
    Allows to run the external program "sphinx-build".
    """

    BUILDER_HTML = 'html'

    BUILDER_LINKCHECK = 'linkcheck'

    BUILDER_SPELLING = 'spelling'

    def __init__(self, build_unit: BuildUnit, module: SphinxModule, builder: str):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        :param builder:     The Sphinx builder to be used
        """
        super().__init__('sphinx-build', '--builder', builder, module.root_directory,
                         path.join(module.output_directory, 'html'))
        self.module = module
        self.builder = builder
        self.print_arguments(True)
        self.install_program(False)
        self.set_build_unit(build_unit)
        self.add_dependencies(
            'furo',
            'myst-parser',
            'sphinx',
            'sphinx-copybutton',
            'sphinx-favicon',
            'sphinx-inline-tabs',
            'sphinxcontrib-spelling',
            'sphinxext-opengraph',
        )

    def _before(self):
        if self.builder == self.BUILDER_SPELLING:
            delete_files(*self.module.find_spelling_files())

    def _after(self):
        if self.builder == self.BUILDER_SPELLING:
            if self.module.find_spelling_files():
                Log.error('Spelling mistakes have been found in the documentation!')
            else:
                Log.info('No spelling mistakes have been found in the documentation.')
