"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for compiling C++ code.
"""
from compilation.cpp.targets import CompileCpp, InstallCpp, SetupCpp
from compilation.modules import CompilationModule
from dependencies.python import VENV
from util.languages import Language
from util.paths import Project
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

SETUP_CPP = 'setup_cpp'

COMPILE_CPP = 'compile_cpp'

INSTALL_CPP = 'install_cpp'

TARGETS = TargetBuilder(BuildUnit('compilation', 'cpp')) \
    .add_build_target(SETUP_CPP) \
        .depends_on(VENV) \
        .set_runnables(SetupCpp()) \
    .add_phony_target(COMPILE_CPP) \
        .depends_on(SETUP_CPP) \
        .set_runnables(CompileCpp()) \
    .add_build_target(INSTALL_CPP) \
        .depends_on(COMPILE_CPP) \
        .set_runnables(InstallCpp()) \
    .build()

MODULES = [
    CompilationModule(
        language=Language.CPP,
        root_directory=Project.Cpp.root_directory,
        build_directory_name=Project.Cpp.build_directory_name,
        install_directory=Project.Python.root_directory,
        installed_file_search=Project.Cpp.file_search() \
            .filter_by_substrings(starts_with='lib', contains='.so') \
            .filter_by_substrings(ends_with='.dylib') \
            .filter_by_substrings(starts_with='mlrl', ends_with='.lib') \
            .filter_by_substrings(ends_with='.dll'),
    ),
]
