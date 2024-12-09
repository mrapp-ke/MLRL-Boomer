"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling C++ code.
"""
from functools import reduce
from typing import List

from compilation.build_options import BuildOptions, EnvBuildOption
from compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from compilation.modules import CompilationModule
from util.files import FileType
from util.log import Log
from util.modules import ModuleRegistry
from util.targets import BuildTarget, PhonyTarget
from util.units import BuildUnit

MODULE_FILTER = CompilationModule.Filter(FileType.cpp())

BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects')) \
        .add(EnvBuildOption(name='test_support', subpackage='common')) \
        .add(EnvBuildOption(name='multi_threading_support', subpackage='common')) \
        .add(EnvBuildOption(name='gpu_support', subpackage='common'))


class SetupCpp(BuildTarget.Runnable):
    """
    Sets up the build system for compiling C++ code.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(MODULE_FILTER):
            MesonSetup(build_unit, module, build_options=BUILD_OPTIONS).run()

    def get_output_files(self, modules: ModuleRegistry) -> List[str]:
        return [module.build_directory for module in modules.lookup(MODULE_FILTER)]

    def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
        Log.info('Removing C++ build files...')
        return super().get_clean_files(modules)


class CompileCpp(PhonyTarget.Runnable):
    """
    Compiles C++ code.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        Log.info('Compiling C++ code...')

        for module in modules.lookup(MODULE_FILTER):
            MesonConfigure(build_unit, module, BUILD_OPTIONS).run()
            MesonCompile(build_unit, module).run()


class InstallCpp(BuildTarget.Runnable):
    """
    Installs shared libraries into the source tree.
    """

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        Log.info('Installing shared libraries into source tree...')

        for module in modules.lookup(MODULE_FILTER):
            MesonInstall(build_unit, module).run()

    def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
        Log.info('Removing shared libraries from source tree...')
        compilation_modules = modules.lookup(MODULE_FILTER)
        return reduce(lambda aggr, module: aggr + module.find_installed_files(), compilation_modules, [])
