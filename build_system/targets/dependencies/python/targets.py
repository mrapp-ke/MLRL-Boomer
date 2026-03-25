"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""

from functools import reduce
from typing import cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.log import Log
from util.package_manager import PackageManager
from util.requirements import RequirementsFile, RequirementsFiles

from targets.dependencies.python.dependencies import DependencyUpdater
from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.table import Table


class InstallPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type that are required by the project's source code.
    """

    def __init__(self, dependency_type: DependencyType | None = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        dependency_modules = (cast(PythonDependencyModule, module) for module in modules)
        requirements_files: list[RequirementsFile] = []
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type),
            dependency_modules,
            requirements_files,
        )

        dependency_type = self.dependency_type

        if dependency_type:
            if dependency_type == DependencyType.BUILD_TIME:
                Log.info('Installing all build-time dependencies...')
            else:
                Log.info('Installing all runtime dependencies...')
        else:
            Log.info('Installing all dependencies...')

        PackageManager.install_all_packages(RequirementsFiles(*requirements_files))


class CheckPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type and checks for outdated ones.
    """

    def __init__(self, dependency_type: DependencyType | None = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        Log.info('Checking for outdated dependencies...')
        dependency_modules = (cast(PythonDependencyModule, module) for module in modules)
        requirements_files: list[RequirementsFile] = []
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type),
            dependency_modules,
            requirements_files,
        )
        outdated_dependencies = DependencyUpdater(*requirements_files).list_outdated_dependencies(build_unit)

        if outdated_dependencies:
            table = Table(build_unit, 'Dependency', 'Declaring file', 'Current version', 'Latest version')

            for outdated_dependency in outdated_dependencies:
                table.add_row(
                    str(outdated_dependency.package),
                    str(outdated_dependency.requirements_file),
                    str(outdated_dependency.outdated),
                    outdated_dependency.latest.min_version,
                )

            table.sort_rows(0, 1)
            Log.info(f'The following dependencies are outdated:\n\n{table}')
        else:
            Log.info('All dependencies are up-to-date!')


class UpdatePythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type and updates outdated ones.
    """

    def __init__(self, dependency_type: DependencyType | None = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        Log.info('Updating outdated dependencies...')
        dependency_modules = (cast(PythonDependencyModule, module) for module in modules)
        requirements_files: list[RequirementsFile] = []
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type),
            dependency_modules,
            requirements_files,
        )
        updated_dependencies = DependencyUpdater(*requirements_files).update_outdated_dependencies(build_unit)

        if updated_dependencies:
            table = Table(build_unit, 'Dependency', 'Declaring file', 'Previous version', 'Updated version')

            for updated_dependency in updated_dependencies:
                table.add_row(
                    str(updated_dependency.package),
                    str(updated_dependency.requirements_file),
                    str(updated_dependency.outdated),
                    str(updated_dependency.latest),
                )

            table.sort_rows(0, 1)
            Log.info(f'The following dependencies have been updated:\n\n{table}')
        else:
            Log.info('No dependencies must be updated!')
