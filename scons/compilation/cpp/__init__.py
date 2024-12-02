"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for compiling C++ code.
"""
from compilation.cpp.targets import CompileCpp, InstallCpp, SetupCpp
from dependencies.python import VENV
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
