"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""
from functools import reduce
from typing import List

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
            table = Table(build_unit, 'Dependency', 'Installed version', 'Latest version')

            for outdated_dependency in outdated_dependencies:
                table.add_row(str(outdated_dependency.installed.package), outdated_dependency.installed.version,
                              outdated_dependency.latest.version)

            table.sort_rows(0, 1)
            Log.info('The following dependencies are outdated:\n\n%s', str(table))
        else:
            Log.info('All dependencies are up-to-date!')
