"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing Python dependencies that are required by the project.
"""

from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.dependencies.python.modules import DependencyType, PythonDependencyModule
from targets.dependencies.python.python import update_python_version
from targets.project import Project

TARGETS = (
    TargetBuilder(BuildUnit.for_file(Path(__file__)))
    .add_phony_target('update_python_version')
    .set_functions(update_python_version)
    .build()
)

MODULES = [
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory=Project.BuildSystem.root_directory,
        requirements_file_search=Project.BuildSystem.file_search(),
    ),
] + [
    PythonDependencyModule(
        dependency_type=DependencyType.RUNTIME,
        root_directory=subproject,
        requirements_file_search=Project.Python.file_search(),
    )
    for subproject in Project.Python.find_subprojects()
]
