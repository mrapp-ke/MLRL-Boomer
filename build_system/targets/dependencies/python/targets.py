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

from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.python.pip import PipList
from targets.dependencies.table import Table


class InstallRuntimeDependencies(PhonyTarget.Runnable):
    """
    Installs all runtime dependencies that are required by the project's source code.
    """

    def __init__(self):
        super().__init__(PythonDependencyModule.Filter(DependencyType.RUNTIME))

    def run_all(self, _: BuildUnit, modules: List[Module]):
        requirements_files = reduce(lambda aggr, module: aggr + module.find_requirements_files(), modules, [])
        PipList(*requirements_files).install_all_packages()


class CheckPythonDependencies(PhonyTarget.Runnable):
    """
    Installs all Python dependencies used by the project and checks for outdated ones.
    """

    def __init__(self):
        super().__init__(PythonDependencyModule.Filter())

    def run_all(self, build_unit: BuildUnit, modules: List[Module]):
        requirements_files = reduce(lambda aggr, module: aggr + module.find_requirements_files(), modules, [])
        pip = PipList(*requirements_files)
        Log.info('Installing all dependencies...')
        pip.install_all_packages()
        Log.info('Checking for outdated dependencies...')
        outdated_dependencies = pip.list_outdated_dependencies()

        if outdated_dependencies:
            table = Table(build_unit, 'Dependency', 'Requirements file', 'Installed version', 'Latest version')

            for outdated_dependency in outdated_dependencies:
                table.add_row(str(outdated_dependency.package), outdated_dependency.requirements_file,
                              outdated_dependency.outdated.min_version, outdated_dependency.latest.min_version)

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
        :param dependency_type: The type of the Python dependencies to be updated or None, if all dependencies should be
                                updated
        """
        super().__init__(
            PythonDependencyModule.Filter(dependency_type) if dependency_type else PythonDependencyModule.Filter())
        self.dependency_type = dependency_type

    def run_all(self, build_unit: BuildUnit, modules: List[Module]):
        requirements_files = reduce(lambda aggr, module: aggr + module.find_requirements_files(), modules, [])
        pip = PipList(*requirements_files)
        Log.info('Installing %s dependencies...',
                 ('all build-time' if self.dependency_type == DependencyType.BUILD_TIME else 'all runtime')
                 if self.dependency_type else 'all')
        pip.install_all_packages()
        Log.info('Updating outdated dependencies...')
        updated_dependencies = pip.update_outdated_dependencies()

        if updated_dependencies:
            table = Table(build_unit, 'Dependency', 'Requirements file', 'Previous version', 'Updated version')
            for updated_dependency in updated_dependencies:
                table.add_row(str(updated_dependency.package), updated_dependency.requirements_file,
                              str(updated_dependency.outdated), str(updated_dependency.latest))

            table.sort_rows(0, 1)
            Log.info('The following dependencies have been updated:\n\n%s', str(table))
        else:
            Log.info('No dependencies must be updated!')
