"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling C++ code.
"""
from os import path
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

MODULE_FILTER = CompilationModule.Filter(FileType.cpp())

BUILD_OPTIONS = BuildOptions() \
        .add(ConstantBuildOption('cpp_std', Project.Cpp.cpp_version(),
                                 *[path.basename(subproject) for subproject in Project.Cpp.find_subprojects()])) \
        .add(EnvBuildOption(SubprojectModule.ENV_SUBPROJECTS.lower())) \
        .add(EnvBuildOption('buildtype', default_value='release')) \
        .add(EnvBuildOption('test_support', 'common')) \
        .add(EnvBuildOption('multi_threading_support', 'common')) \
        .add(EnvBuildOption('gpu_support', 'common'))

MESON_OPTIONS = ['-D', 'library_version=' + str(Project.version(release=True))]


class SetupCpp(BuildTarget.Runnable):
    """
    Sets up the build system for compiling C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        MesonSetup(build_unit, compilation_module, *MESON_OPTIONS, build_options=BUILD_OPTIONS).run()

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        compilation_module = cast(CompilationModule, module)
        return [compilation_module.build_directory]

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        compilation_module = cast(CompilationModule, module)
        Log.info('Removing C++ build files from directory "%s"...', compilation_module.root_directory)
        return super().get_clean_files(build_unit, compilation_module)


class CompileCpp(PhonyTarget.Runnable):
    """
    Compiles C++ code.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        Log.info('Compiling C++ code in directory "%s"...', compilation_module.root_directory)
        MesonConfigure(build_unit, compilation_module, *MESON_OPTIONS, build_options=BUILD_OPTIONS).run()
        MesonCompile(build_unit, compilation_module).run()


class InstallCpp(BuildTarget.Runnable):
    """
    Installs shared libraries into the source tree.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        compilation_module = cast(CompilationModule, module)
        Log.info('Installing shared libraries from directory "%s" into source tree...',
                 compilation_module.root_directory)
        MesonInstall(build_unit, compilation_module).run()

    @override
    def get_clean_files(self, _: BuildUnit, module: Module) -> List[str]:
        compilation_module = cast(CompilationModule, module)
        Log.info('Removing shared libraries installed from directory "%s" from source tree...',
                 compilation_module.root_directory)
        return compilation_module.find_installed_files()
