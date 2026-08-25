"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for updating Meson wrap files requirements that declare runtime dependencies required by the
project's C++ code.
"""

from functools import reduce
from typing import override, cast

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from targets.dependencies.cpp.dependencies import WrapFileUpdater
from targets.dependencies.cpp.modules import WrapFileModule
from targets.dependencies.cpp.wrap_file import WrapFile
from targets.dependencies.table import Table
from util.log import Log


class UpdateWrapFiles(PhonyTarget.Runnable):
    """
    Updates outdates dependencies declared in Meson wrap files.
    """

    def __init__(self):
        super().__init__(WrapFileModule.Filter())

    @override
    def run_all(self, build_unit: BuildUnit, modules: list[Module]):
        Log.info('Updating outdated dependencies in Meson wrap files...')
        dependency_modules = (cast(WrapFileModule, module) for module in modules)
        wrap_files: list[WrapFile] = []
        wrap_files = reduce(
            lambda aggr, module: aggr + module.find_wrap_files(build_unit),
            dependency_modules,
            wrap_files,
        )
        updated_dependencies = WrapFileUpdater(*wrap_files).update_outdated_dependencies(build_unit)

        if updated_dependencies:
            table = Table(build_unit, 'Dependency', 'Declaring file', 'Previous version', 'Updated version')

            for updated_dependency in updated_dependencies:
                table.add_row(
                    str(updated_dependency.wrap_file.dependency_name),
                    str(updated_dependency.wrap_file),
                    str(updated_dependency.outdated),
                    str(updated_dependency.latest),
                )

            table.sort_rows(0, 1)
            Log.info(f'The following dependencies in Meson wrap files have been updated:\n\n{table}')
        else:
            Log.info('No dependencies in Meson wrap files must be updated!')
