"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for compiling native library dependencies on macOS. This is necessary when building
pre-built packages via "cibuildwheel", because the de facto package manager "homebrew" will install libraries that are
compiled for the specific macOS version the build machine is running, instead of using the proper version for the target
platform (see https://cibuildwheel.pypa.io/en/stable/faq/#missing-dependencies).
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder

from targets.dependencies.macos.targets import compile_libomp

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target('dependency_libomp').set_functions(compile_libomp) \
    .build()
