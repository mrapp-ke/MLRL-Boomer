"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for building and install Python wheel packages.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.compilation import INSTALL
from targets.packaging.modules import PythonPackageModule
from targets.packaging.targets import BuildPythonWheels, GeneratePyprojectTomlFiles, InstallPythonWheels
from targets.project import Project

PYPROJECT_TOML = 'pyproject_toml'

BUILD_WHEELS = 'build_wheels'

INSTALL_WHEELS = 'install_wheels'

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_build_target(PYPROJECT_TOML) \
        .depends_on(INSTALL) \
        .set_runnables(GeneratePyprojectTomlFiles()) \
    .add_build_target(BUILD_WHEELS) \
        .depends_on(PYPROJECT_TOML, clean_dependencies=True) \
        .set_runnables(BuildPythonWheels()) \
    .add_build_target(INSTALL_WHEELS) \
        .depends_on(BUILD_WHEELS) \
        .set_runnables(InstallPythonWheels()) \
    .build()

MODULES = [
    PythonPackageModule(
        root_directory=subproject,
        wheel_directory_name=Project.Python.wheel_directory_name,
    ) for subproject in Project.Python.find_subprojects()
]
