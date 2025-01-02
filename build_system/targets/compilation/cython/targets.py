"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling Cython code.
"""
from typing import List

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget, PhonyTarget
from util.files import FileType
from util.log import Log

from targets.compilation.build_options import BuildOptions, EnvBuildOption
from targets.compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from targets.compilation.modules import CompilationModule
from targets.modules import SubprojectModule

MODULE_FILTER = CompilationModule.Filter(FileType.cython())

BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name=SubprojectModule.ENV_SUBPROJECTS.lower()))


class SetupCython(BuildTarget.Runnable):
    """
    Sets up the build system for compiling the Cython code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        MesonSetup(build_unit, module, build_options=BUILD_OPTIONS) \
            .add_dependencies('cython') \
            .run()

    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        return [module.build_directory]

    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        Log.info('Removing Cython build files from directory "%s"...', module.root_directory)
        return super().get_clean_files(build_unit, module)


class CompileCython(PhonyTarget.Runnable):
    """
    Compiles the Cython code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Compiling Cython code in directory "%s"...', module.root_directory)
        MesonConfigure(build_unit, module, build_options=BUILD_OPTIONS).run()
        MesonCompile(build_unit, module).run()


class InstallCython(BuildTarget.Runnable):
    """
    Installs extension modules into the source tree.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Installing extension modules from directory "%s" into source tree...', module.root_directory)
        MesonInstall(build_unit, module).run()

    def get_clean_files(self, _: BuildUnit, module: Module) -> List[str]:
        Log.info('Removing extension modules installed from directory "%s" from source tree...', module.root_directory)
        return module.find_installed_files()
