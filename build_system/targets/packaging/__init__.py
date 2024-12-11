"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for building and install Python wheel packages.
"""
from os import path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.compilation import INSTALL
from targets.packaging.modules import PythonPackageModule
from targets.packaging.targets import BuildPythonWheels, InstallPythonWheels
from targets.paths import Project

BUILD_WHEELS = 'build_wheels'

INSTALL_WHEELS = 'install_wheels'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_build_target(BUILD_WHEELS) \
        .depends_on(INSTALL) \
        .set_runnables(BuildPythonWheels()) \
    .add_build_target(INSTALL_WHEELS) \
        .depends_on(BUILD_WHEELS) \
        .set_runnables(InstallPythonWheels()) \
    .build()

MODULES = [
    PythonPackageModule(
        root_directory=path.dirname(setup_file),
        wheel_directory_name=Project.Python.wheel_directory_name,
    ) for setup_file in Project.Python.file_search().filter_by_name('setup.py').list(Project.Python.root_directory)
]
