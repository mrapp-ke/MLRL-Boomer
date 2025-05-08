"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing Python dependencies that are required by the project.
"""
from os import path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.python.python import check_python_version, update_python_version
from targets.dependencies.python.targets import CheckPythonDependencies, InstallPythonDependencies, \
    UpdatePythonDependencies
from targets.project import Project

VENV = 'venv'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(VENV).set_runnables(InstallPythonDependencies(DependencyType.RUNTIME)) \
    .add_phony_target('check_runtime_dependencies').set_runnables(CheckPythonDependencies(DependencyType.RUNTIME)) \
    .add_phony_target('check_build_dependencies').set_runnables(CheckPythonDependencies(DependencyType.BUILD_TIME)) \
    .add_phony_target('check_dependencies').set_runnables(CheckPythonDependencies()) \
    .add_phony_target('check_python_version').set_functions(check_python_version) \
    .add_phony_target('update_runtime_dependencies').set_runnables(UpdatePythonDependencies(DependencyType.RUNTIME)) \
    .add_phony_target('update_build_dependencies').set_runnables(UpdatePythonDependencies(DependencyType.BUILD_TIME)) \
    .add_phony_target('update_dependencies').set_runnables(UpdatePythonDependencies()) \
    .add_phony_target('update_python_version').set_functions(update_python_version) \
    .build()

MODULES = [
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory=Project.BuildSystem.root_directory,
        requirements_file_search=Project.BuildSystem.file_search(),
    ),
] + [
    PythonDependencyModule(dependency_type=DependencyType.RUNTIME,
                           root_directory=subproject,
                           requirements_file_search=Project.Python.file_search())
    for subproject in Project.Python.find_subprojects()
]
