"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for YAML files.
"""
from code_style.modules import CodeModule
from code_style.yaml.targets import CheckYamlCodeStyle, EnforceYamlCodeStyle
from util.files import FileSearch
from util.languages import Language
from util.paths import Project
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_YAML = 'format_yaml'

TEST_FORMAT_YAML = 'test_format_yaml'

TARGETS = TargetBuilder(BuildUnit('code_style', 'yaml')) \
    .add_phony_target(FORMAT_YAML).set_runnables(EnforceYamlCodeStyle()) \
    .add_phony_target(TEST_FORMAT_YAML).set_runnables(CheckYamlCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        language=Language.YAML,
        root_directory=Project.root_directory,
        source_file_search=FileSearch().set_recursive(False).set_hidden(True),
    ),
    CodeModule(
        language=Language.YAML,
        root_directory=Project.Github.root_directory,
    ),
]
