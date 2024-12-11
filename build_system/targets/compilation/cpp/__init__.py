"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for compiling C++ code.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileType

from targets.compilation.cpp.targets import CompileCpp, InstallCpp, SetupCpp
from targets.compilation.modules import CompilationModule
from targets.dependencies.python import VENV
from targets.paths import Project

SETUP_CPP = 'setup_cpp'

COMPILE_CPP = 'compile_cpp'

INSTALL_CPP = 'install_cpp'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
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
        file_type=FileType.cpp(),
        root_directory=Project.Cpp.root_directory,
        build_directory_name=Project.Cpp.build_directory_name,
        install_directory=Project.Python.root_directory,
        installed_file_search=Project.Cpp.file_search().filter_by_file_type(FileType.shared_library()),
    ),
]
