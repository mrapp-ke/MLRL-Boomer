"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to build wheel packages via the external program "build".
"""
from core.build_unit import BuildUnit
from util.run import PythonModule

from targets.packaging.modules import PythonPackageModule


class Build(PythonModule):
    """
    Allows to run the external program "build".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonPackageModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('build', '--wheel', module.root_directory)
        self.print_arguments(True)
        self.add_dependencies('wheel')
        self.set_build_unit(build_unit)
