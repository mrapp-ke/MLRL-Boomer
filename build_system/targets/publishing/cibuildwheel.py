"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cibuildwheel".
"""
from pathlib import Path

from core.build_unit import BuildUnit
from util.run import Program


class Cibuildwheel(Program):
    """
    Allows to run the external program "cibuildwheel" to build a Python package.
    """

    def __init__(self, build_unit: BuildUnit, package_directory: Path, print_build_identifiers: bool = False):
        """
        :param build_unit:          The build unit from which the program should be run
        :param package_directory:   The path to the directory of the Python package to be built
        """
        super().__init__('cibuildwheel', str(package_directory))
        self.set_build_unit(build_unit)
        self.add_conditional_arguments(print_build_identifiers, '--print-build-identifiers')
