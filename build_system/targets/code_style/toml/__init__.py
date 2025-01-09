"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for TOML files.
"""
from core.build_unit import BuildUnit
from core.targets import TargetBuilder
from util.files import FileSearch, FileType

from targets.code_style.modules import CodeModule
from targets.code_style.toml.targets import CheckTomlCodeStyle, EnforceTomlCodeStyle
from targets.paths import Project

FORMAT_TOML = 'format_toml'

TEST_FORMAT_TOML = 'test_format_toml'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(FORMAT_TOML).set_runnables(EnforceTomlCodeStyle()) \
    .add_phony_target(TEST_FORMAT_TOML).set_runnables(CheckTomlCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.toml(),
        root_directory=Project.BuildSystem.root_directory,
        source_file_search=FileSearch().set_recursive(True).set_hidden(True),
    ),
    CodeModule(file_type=FileType.toml(),
               root_directory=Project.Python.root_directory,
               source_file_search=Project.Python.file_search()),
]
