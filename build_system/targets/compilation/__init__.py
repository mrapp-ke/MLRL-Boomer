"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for compiling code.
"""
from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.compilation.cpp import COMPILE_CPP, INSTALL_CPP
from targets.compilation.cython import COMPILE_CYTHON, INSTALL_CYTHON

INSTALL = 'install'

TARGETS = TargetBuilder(BuildUnit('targets', 'compilation')) \
    .add_phony_target('compile') \
        .depends_on(COMPILE_CPP, COMPILE_CYTHON, clean_dependencies=True) \
        .nop() \
    .add_phony_target(INSTALL) \
        .depends_on(INSTALL_CPP, INSTALL_CYTHON, clean_dependencies=True) \
        .nop() \
    .build()
