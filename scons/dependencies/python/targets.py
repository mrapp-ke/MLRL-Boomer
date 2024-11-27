"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""
from dependencies.python.modules import DependencyType, PythonDependencyModule
from util.modules import ModuleRegistry
from util.pip import Pip
from util.targets import PhonyTarget
from util.units import BuildUnit


class InstallRuntimeDependencies(PhonyTarget.Runnable):
    """
    Installs all runtime dependencies that are required by the project's source code.
    """

    def run(self, _: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(PythonDependencyModule.Filter(DependencyType.RUNTIME)):
            for requirements_file in module.find_requirements_files():
                Pip(requirements_file).install_all_packages()
