"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for checking and enforcing code style definitions for YAML files.
"""
from code_style.yaml.targets import CheckYamlCodeStyle, EnforceYamlCodeStyle
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_YAML = 'format_yaml'

TEST_FORMAT_YAML = 'test_format_yaml'

TARGETS = TargetBuilder(BuildUnit.by_name('code_style', 'yaml')) \
    .add_phony_target(FORMAT_YAML).set_runnables(EnforceYamlCodeStyle()) \
    .add_phony_target(TEST_FORMAT_YAML).set_runnables(CheckYamlCodeStyle()) \
    .build()
