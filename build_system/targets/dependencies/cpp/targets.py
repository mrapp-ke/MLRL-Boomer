"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for installing runtime requirements that are required by the project's source code.
"""

from functools import reduce
from typing import cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from targets.dependencies.cpp.modules import WrapFileModule
from targets.dependencies.cpp.wrap_file import WrapFile
from targets.dependencies.python.dependencies import DependencyUpdater
from targets.dependencies.table import Table
from util.log import Log


class CheckWrapFiles(PhonyTarget.Runnable):
    """
    Cheks for outdated dependencies declared in Meson wrap files.
    """

    def __init__(self):
        super().__init__(WrapFileModule.Filter())

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        Log.info('Checking for outdated dependencies...')
        dependency_modules = (cast(WrapFileModule, module) for module in modules)

        wrap_files: list[WrapFile] = []
        wrap_files = reduce(
            lambda aggr, module: aggr + module.find_wrap_files(build_unit),
            dependency_modules,
            wrap_files,
        )
        outdated_dependencies = DependencyUpdater(*wrap_files).list_outdated_dependencies(build_unit)

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


class UpdateWrapFiles(PhonyTarget.Runnable):
    """
    Updates the versions of dependencies declared in Meson wrap files.
    """

    def __init__(self):
        super().__init__(WrapFileModule.Filter())

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        Log.info('Updating outdated dependencies...')
        dependency_modules = (cast(WrapFileModule, module) for module in modules)
        wrap_files: list[WrapFile] = []
        wrap_files = reduce(
            lambda aggr, module: aggr + module.find_wrap_files(build_unit),
            dependency_modules,
            wrap_files,
        )
        updated_dependencies = DependencyUpdater(*wrap_files).update_outdated_dependencies(build_unit)

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
