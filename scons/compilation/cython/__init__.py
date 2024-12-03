"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for compiling Cython code.
"""
from compilation.cpp import COMPILE_CPP
from compilation.cython.targets import CompileCython, InstallCython, SetupCython
from compilation.modules import CompilationModule
from util.files import FileSearch
from util.languages import Language
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

SETUP_CYTHON = 'setup_cython'

COMPILE_CYTHON = 'compile_cython'

INSTALL_CYTHON = 'install_cython'

TARGETS = TargetBuilder(BuildUnit('compilation', 'cython')) \
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
        language=Language.CYTHON,
        root_directory='python',
        installed_file_search=FileSearch() \
            .filter_by_substrings(not_starts_with='lib', ends_with='.so') \
            .filter_by_substrings(ends_with='.pyd') \
            .filter_by_substrings(not_starts_with='mlrl', ends_with='.lib'),
    ),
]
