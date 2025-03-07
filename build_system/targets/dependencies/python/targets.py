"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""
from functools import reduce
from typing import List, Optional

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.log import Log
from util.pip import Pip

from targets.dependencies.python.dependencies import DependencyUpdater
from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.table import Table


class InstallPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type that are required by the project's source code.
    """

    def __init__(self, dependency_type: Optional[DependencyType] = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    def run_all(self, build_unit: BuildUnit, modules: List[Module]):
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type), modules, [])
        pip = Pip(*requirements_files)
        Log.info('Installing %s dependencies...',
                 ('all build-time' if self.dependency_type == DependencyType.BUILD_TIME else 'all runtime')
                 if self.dependency_type else 'all')
        pip.install_all_packages()


class CheckPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type and checks for outdated ones.
    """

    def __init__(self, dependency_type: Optional[DependencyType] = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    def run_all(self, build_unit: BuildUnit, modules: List[Module]):
        Log.info('Checking for outdated dependencies...')
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type), modules, [])
        outdated_dependencies = DependencyUpdater(*requirements_files).list_outdated_dependencies(build_unit)

        if outdated_dependencies:
            table = Table(build_unit, 'Dependency', 'Declaring file', 'Current version', 'Latest version')

            for outdated_dependency in outdated_dependencies:
                table.add_row(str(outdated_dependency.package), str(outdated_dependency.requirements_file),
                              str(outdated_dependency.outdated), outdated_dependency.latest.min_version)

            table.sort_rows(0, 1)
            Log.info('The following dependencies are outdated:\n\n%s', str(table))
        else:
            Log.info('All dependencies are up-to-date!')


class UpdatePythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies of a specific type and updates outdated ones.
    """

    def __init__(self, dependency_type: Optional[DependencyType] = None):
        """
        :param dependency_type: The type of the Python dependencies to be installed or None, if all dependencies should
                                be installed
        """
        super().__init__(PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    def run_all(self, build_unit: BuildUnit, modules: List[Module]):
        Log.info('Updating outdated dependencies...')
        requirements_files = reduce(
            lambda aggr, module: aggr + module.find_requirements_files(build_unit, self.dependency_type), modules, [])
        updated_dependencies = DependencyUpdater(*requirements_files).update_outdated_dependencies(build_unit)

        if updated_dependencies:
            table = Table(build_unit, 'Dependency', 'Declaring file', 'Previous version', 'Updated version')

            for updated_dependency in updated_dependencies:
                table.add_row(str(updated_dependency.package), str(updated_dependency.requirements_file),
                              str(updated_dependency.outdated), str(updated_dependency.latest))

            table.sort_rows(0, 1)
            Log.info('The following dependencies have been updated:\n\n%s', str(table))
        else:
            Log.info('No dependencies must be updated!')
