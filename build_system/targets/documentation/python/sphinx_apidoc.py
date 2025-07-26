"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "sphinx-apidoc".
"""
from typing import override

from core.build_unit import BuildUnit
from util.files import FileType
from util.io import create_directories, delete_files
from util.run import Program

from targets.documentation.python.modules import PythonApidocModule


class SphinxApidoc(Program):
    """
    Allows to run the external program "sphinx-apidoc".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonApidocModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('sphinx-apidoc', '--separate', '--module-first', '--no-toc', '-o',
                         str(module.output_directory), str(module.source_directory),
                         *['*.' + suffix + '*' for suffix in FileType.extension_module().suffixes],
                         *['*.' + suffix + '*' for suffix in FileType.shared_library().suffixes])
        self.module = module
        self.print_arguments(True)
        self.install_program(False)
        self.add_dependencies('sphinx')
        self.set_build_unit(build_unit)

    @override
    def _before(self):
        create_directories(self.module.output_directory)

    @override
    def _after(self):
        root_rst_file = self.module.output_directory / (self.module.source_directory.name + '.rst')
        delete_files(root_rst_file, accept_missing=False)
