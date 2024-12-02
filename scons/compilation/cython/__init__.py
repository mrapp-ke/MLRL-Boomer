"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for compiling C++ code.
"""
from compilation.cpp import COMPILE_CPP
from compilation.cython.targets import CompileCython, InstallCython, SetupCython
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
