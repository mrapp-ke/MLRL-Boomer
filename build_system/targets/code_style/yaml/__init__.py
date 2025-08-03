"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for YAML files.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder
from util.files import FileSearch, FileType

from targets.code_style.modules import CodeModule
from targets.code_style.yaml.targets import CheckYamlCodeStyle, EnforceYamlCodeStyle
from targets.project import Project

FORMAT_YAML = 'format_yaml'

TEST_FORMAT_YAML = 'test_format_yaml'

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target(FORMAT_YAML).set_runnables(EnforceYamlCodeStyle()) \
    .add_phony_target(TEST_FORMAT_YAML).set_runnables(CheckYamlCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.yaml(),
        root_directory=Project.Readthedocs.root_directory,
        source_file_search=FileSearch().set_recursive(False).set_hidden(True),
    ),
    CodeModule(
        file_type=FileType.yaml(),
        root_directory=Project.Github.root_directory,
    ),
    CodeModule(
        file_type=FileType.yaml(),
        root_directory=Project.Python.root_directory,
    ),
]
