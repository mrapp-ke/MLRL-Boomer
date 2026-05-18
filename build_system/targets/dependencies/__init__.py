"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing dependencies that are required by the project.
"""

from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder
from targets.dependencies.cpp.targets import UpdateWrapFiles
from targets.dependencies.python.modules import DependencyType
from targets.dependencies.python.targets import (
    UpdatePythonDependencies,
    InstallPythonDependencies,
)

VENV = 'venv'

INSTALL_RUNTIME_DEPENDENCIES = 'install_runtime_dependencies'

TARGETS = (
    TargetBuilder(BuildUnit.for_file(Path(__file__)))
    .add_phony_target(VENV)
    .nop()
    .add_phony_target(INSTALL_RUNTIME_DEPENDENCIES)
    .depends_on(VENV)
    .set_runnables(InstallPythonDependencies(DependencyType.RUNTIME))
    .add_phony_target('update_runtime_dependencies')
    .set_runnables(UpdatePythonDependencies(DependencyType.RUNTIME), UpdateWrapFiles())
    .add_phony_target('update_build_dependencies')
    .set_runnables(UpdatePythonDependencies(DependencyType.BUILD_TIME))
    .add_phony_target('update_dependencies')
    .set_runnables(UpdatePythonDependencies())
    .build()
)
