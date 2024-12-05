"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling Cython code.
"""
from functools import reduce
from typing import List

from compilation.build_options import BuildOptions, EnvBuildOption
from compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from compilation.modules import CompilationModule
from util.files import FileType
from util.modules import ModuleRegistry
from util.targets import BuildTarget, PhonyTarget
from util.units import BuildUnit

MODULE_FILTER = CompilationModule.Filter(FileType.cython())

BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects'))


class SetupCython(BuildTarget.Runnable):
    """
    Sets up the build system for compiling the Cython code.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            MesonSetup(build_unit, module) \
                .add_dependencies('cython') \
                .run()

    def get_output_files(self, modules: ModuleRegistry) -> List[str]:
        return [module.build_directory for module in modules.lookup(MODULE_FILTER)]

    def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
        print('Removing Cython build files...')
        return super().get_clean_files(modules)


class CompileCython(PhonyTarget.Runnable):
    """
    Compiles the Cython code.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        print('Compiling Cython code...')

        for module in modules.lookup(MODULE_FILTER):
            MesonConfigure(build_unit, module, build_options=BUILD_OPTIONS)
            MesonCompile(build_unit, module).run()


class InstallCython(BuildTarget.Runnable):
    """
    Installs extension modules into the source tree.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        print('Installing extension modules into source tree...')

        for module in modules.lookup(MODULE_FILTER):
            MesonInstall(build_unit, module).run()

    def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
        print('Removing extension modules from source tree...')
        compilation_modules = modules.lookup(MODULE_FILTER)
        return reduce(lambda aggr, module: aggr + module.find_installed_files(), compilation_modules, [])
