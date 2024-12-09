"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for checking and enforcing code style definitions.
"""
from code_style.cpp import FORMAT_CPP, TEST_FORMAT_CPP
from code_style.markdown import FORMAT_MARKDOWN, TEST_FORMAT_MARKDOWN
from code_style.python import FORMAT_PYTHON, TEST_FORMAT_PYTHON
from code_style.yaml import FORMAT_YAML, TEST_FORMAT_YAML
from util.targets import TargetBuilder
from util.units import BuildUnit

TARGETS = TargetBuilder(BuildUnit('code_style')) \
    .add_phony_target('format') \
        .depends_on(FORMAT_PYTHON, FORMAT_CPP, FORMAT_MARKDOWN, FORMAT_YAML) \
        .nop() \
    .add_phony_target('test_format') \
       .depends_on(TEST_FORMAT_PYTHON, TEST_FORMAT_CPP, TEST_FORMAT_MARKDOWN, TEST_FORMAT_YAML) \
       .nop() \
    .build()
