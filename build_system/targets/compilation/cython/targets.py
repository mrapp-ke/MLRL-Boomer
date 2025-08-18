"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling Cython code.
"""
from pathlib import Path
from typing import List, cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget, PhonyTarget
from util.files import FileType
from util.log import Log

from targets.compilation.build_options import BuildOptions, ConstantBuildOption, EnvBuildOption
from targets.compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from targets.compilation.modules import CompilationModule
from targets.modules import SubprojectModule
from targets.project import Project

MODULE_FILTER = CompilationModule.Filter(FileType.cython())

BUILD_OPTIONS = BuildOptions() \
        .add(ConstantBuildOption('cpp_std', Project.Cpp.cpp_version(),
                                 *[subproject.name for subproject in Project.Cpp.find_subprojects()])) \
        .add(EnvBuildOption(SubprojectModule.ENV_SUBPROJECTS.lower())) \
        .add(EnvBuildOption('buildtype', default_value='release'))


class SetupCython(BuildTarget.Runnable):
    """
    Sets up the build system for compiling the Cython code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        MesonSetup(build_unit, compilation_module, build_options=BUILD_OPTIONS) \
            .add_dependencies('cython') \
            .run()

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[Path]:
        compilation_module = cast(CompilationModule, module)
        return [compilation_module.build_directory]

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[Path]:
        compilation_module = cast(CompilationModule, module)
        Log.info('Removing Cython build files from directory "%s"...', compilation_module.root_directory)
        return super().get_clean_files(build_unit, compilation_module)


class CompileCython(PhonyTarget.Runnable):
    """
    Compiles the Cython code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        Log.info('Compiling Cython code in directory "%s"...', compilation_module.root_directory)
        MesonSetup(build_unit, compilation_module, '--reconfigure', build_options=BUILD_OPTIONS).run()
        MesonConfigure(build_unit, compilation_module, build_options=BUILD_OPTIONS).run()
        MesonCompile(build_unit, compilation_module).run()


class InstallCython(BuildTarget.Runnable):
    """
    Installs extension modules into the source tree.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        Log.info('Installing extension modules from directory "%s" into source tree...',
                 compilation_module.root_directory)
        MesonInstall(build_unit, compilation_module).run()

    @override
    def get_clean_files(self, _: BuildUnit, module: Module) -> List[Path]:
        compilation_module = cast(CompilationModule, module)
        Log.info('Removing extension modules installed from directory "%s" from source tree...',
                 compilation_module.root_directory)
        return compilation_module.find_installed_files()
