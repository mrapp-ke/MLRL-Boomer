"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for building and install Python wheel packages.
"""
from os import path

from compilation import INSTALL
from packaging.modules import PythonPackageModule
from packaging.targets import BuildPythonWheels, InstallPythonWheels
from util.paths import Project
from util.targets import TargetBuilder
from util.units import BuildUnit

BUILD_WHEELS = 'build_wheels'

INSTALL_WHEELS = 'install_wheels'

MODULES = [
    PythonPackageModule(
        root_directory=path.dirname(setup_file),
        wheel_directory_name=Project.Python.wheel_directory_name,
    ) for setup_file in Project.Python.file_search().filter_by_name('setup.py').list(Project.Python.root_directory)
]

TARGETS = TargetBuilder(BuildUnit('packaging')) \
    .add_phony_target(BUILD_WHEELS) \
        .depends_on(INSTALL) \
        .depends_on_build_targets(
            MODULES,
            lambda module, target_builder: target_builder.set_runnables(BuildPythonWheels(module.root_directory))
        ) \
        .nop() \
    .add_phony_target(INSTALL_WHEELS) \
        .depends_on(BUILD_WHEELS) \
        .depends_on_phony_targets(
            MODULES,
            lambda module, target_builder: target_builder.set_runnables(InstallPythonWheels(module.root_directory))
        ) \
        .nop() \
    .build()
