"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing Python dependencies that are required by the project.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from dependencies.python.modules import DependencyType, PythonDependencyModule
from dependencies.python.targets import CheckPythonDependencies, InstallRuntimeDependencies
from util.paths import Project

VENV = 'venv'

TARGETS = TargetBuilder(BuildUnit('dependencies', 'python')) \
    .add_phony_target(VENV).set_runnables(InstallRuntimeDependencies()) \
    .add_phony_target('check_dependencies').set_runnables(CheckPythonDependencies()) \
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
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory=Project.Documentation.root_directory,
        requirements_file_search=Project.Documentation.file_search(),
    ),
]
