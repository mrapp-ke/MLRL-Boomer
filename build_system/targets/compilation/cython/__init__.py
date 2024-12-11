"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for compiling Cython code.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileType

from targets.compilation.cpp import COMPILE_CPP
from targets.compilation.cython.targets import CompileCython, InstallCython, SetupCython
from targets.compilation.modules import CompilationModule
from targets.paths import Project

SETUP_CYTHON = 'setup_cython'

COMPILE_CYTHON = 'compile_cython'

INSTALL_CYTHON = 'install_cython'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_build_target(SETUP_CYTHON) \
        .depends_on(COMPILE_CPP) \
        .set_runnables(SetupCython()) \
    .add_phony_target(COMPILE_CYTHON) \
        .depends_on(SETUP_CYTHON) \
        .set_runnables(CompileCython()) \
    .add_build_target(INSTALL_CYTHON) \
        .depends_on(COMPILE_CYTHON) \
        .set_runnables(InstallCython()) \
    .build()

MODULES = [
    CompilationModule(
        file_type=FileType.cython(),
        root_directory=Project.Python.root_directory,
        build_directory_name=Project.Python.build_directory_name,
        installed_file_search=Project.Python.file_search().filter_by_file_type(FileType.extension_module()),
    ),
]
