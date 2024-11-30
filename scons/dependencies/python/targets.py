"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""
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
        for module in modules.lookup(PythonDependencyModule.Filter(DependencyType.RUNTIME)):
            for requirements_file in module.find_requirements_files():
                Pip(requirements_file).install_all_packages()


class CheckPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies used by the project and checks for outdated ones.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        outdated_dependencies_by_requirements_file = {}

        for module in modules.lookup(PythonDependencyModule.Filter()):
            for requirements_file in module.find_requirements_files():
                outdated_dependencies = Pip(requirements_file).list_outdated_dependencies()

                if outdated_dependencies:
                    outdated_dependencies_by_requirements_file[requirements_file] = outdated_dependencies

        if outdated_dependencies_by_requirements_file:
            table = Table(build_unit, 'Requirements file', 'Dependency', 'Installed version', 'Latest version')

            for requirements_file, outdated_dependencies in outdated_dependencies_by_requirements_file.items():
                for dependency in outdated_dependencies:
                    table.add_row(requirements_file, str(dependency.installed.package), dependency.installed.version,
                                  dependency.latest.version)

            table.sort_rows(0, 1)
            print('The following dependencies are outdated:\n')
            print(str(table))
        else:
            print('All dependencies are up-to-date!')
