"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to build wheel packages via the external program "build".
"""
from packaging.modules import PythonPackageModule
from util.run import PythonModule
from util.units import BuildUnit


class Build(PythonModule):
    """
    Allows to run the external program "build".
    """

    def __init__(self, build_unit: BuildUnit, module: PythonPackageModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__('build', '--no-isolation', '--wheel', module.root_directory)
        self.print_arguments(True)
        self.add_dependencies('wheel', 'setuptools')
        self.set_build_unit(build_unit)
