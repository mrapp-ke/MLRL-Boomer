"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing Python dependencies that are required by the project.
"""
from dependencies.python.modules import DependencyType, PythonDependencyModule
from dependencies.python.targets import CheckPythonDependencies, InstallRuntimeDependencies
from util.files import FileSearch
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

VENV = 'venv'

TARGETS = TargetBuilder(BuildUnit('dependencies', 'python')) \
    .add_phony_target(VENV).set_runnables(InstallRuntimeDependencies()) \
    .add_phony_target('check_dependencies').set_runnables(CheckPythonDependencies()) \
    .build()

MODULES = [
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory='scons',
        requirements_file_search=FileSearch().set_recursive(True),
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.RUNTIME,
        root_directory='python',
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory='doc',
        requirements_file_search=FileSearch().set_recursive(True),
    ),
]
