"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing Python dependencies that are required by the project.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder

from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.python.targets import CheckPythonDependencies, InstallRuntimeDependencies, \
    UpdatePythonDependencies
from targets.paths import Project

VENV = 'venv'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(VENV).set_runnables(InstallRuntimeDependencies()) \
    .add_phony_target('check_runtime_dependencies').set_runnables(CheckPythonDependencies(DependencyType.RUNTIME)) \
    .add_phony_target('check_build_dependencies').set_runnables(CheckPythonDependencies(DependencyType.BUILD_TIME)) \
    .add_phony_target('check_dependencies').set_runnables(CheckPythonDependencies()) \
    .add_phony_target('update_runtime_dependencies').set_runnables(UpdatePythonDependencies(DependencyType.RUNTIME)) \
    .add_phony_target('update_build_dependencies').set_runnables(UpdatePythonDependencies(DependencyType.BUILD_TIME)) \
    .add_phony_target('update_dependencies').set_runnables(UpdatePythonDependencies()) \
    .build()

MODULES = [
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory=Project.BuildSystem.root_directory,
        requirements_file_search=Project.BuildSystem.file_search(),
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.RUNTIME,
        root_directory=Project.Python.root_directory,
        requirements_file_search=Project.Python.file_search(),
    ),
]
