"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""
from functools import reduce

from dependencies.python.modules import DependencyType, PythonDependencyModule
from util.modules import ModuleRegistry
from util.pip import Pip
from util.table import Table
from util.targets import PhonyTarget
from util.units import BuildUnit


class InstallRuntimeDependencies(PhonyTarget.Runnable):
    """
    Installs all runtime dependencies that are required by the project's source code.
    """

    def run(self, _: BuildUnit, modules: ModuleRegistry):
        dependency_modules = modules.lookup(PythonDependencyModule.Filter(DependencyType.RUNTIME))
        requirements_files = reduce(lambda aggr, module: aggr + module.find_requirements_files(), dependency_modules,
                                    [])
        Pip(*requirements_files).install_all_packages()


class CheckPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies used by the project and checks for outdated ones.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        dependency_modules = modules.lookup(PythonDependencyModule.Filter())
        requirements_files = reduce(lambda aggr, module: aggr + module.find_requirements_files(), dependency_modules,
                                    [])
        outdated_dependencies = Pip(*requirements_files).list_outdated_dependencies()

        if outdated_dependencies:
            table = Table(build_unit, 'Dependency', 'Installed version', 'Latest version')

            for outdated_dependency in outdated_dependencies:
                table.add_row(str(outdated_dependency.installed.package), outdated_dependency.installed.version,
                              outdated_dependency.latest.version)

            table.sort_rows(0, 1)
            print('The following dependencies are outdated:\n')
            print(str(table))
        else:
            print('All dependencies are up-to-date!')
