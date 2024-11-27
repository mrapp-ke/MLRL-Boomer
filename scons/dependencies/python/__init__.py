"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for updating the Python runtime dependencies that are required by the project's source code.
"""
from dependencies.python.targets import InstallRuntimeDependencies
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

VENV = 'venv'

TARGETS = TargetBuilder(BuildUnit('dependencies', 'python')) \
    .add_phony_target(VENV).set_runnables(InstallRuntimeDependencies()) \
    .build()
