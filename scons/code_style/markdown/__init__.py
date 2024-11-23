"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for checking and enforcing code style definitions for Markdown files.
"""
from code_style.markdown.targets import CheckMarkdownCodeStyle, EnforceMarkdownCodeStyle
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_MARKDOWN = 'format_md'

TEST_FORMAT_MARKDOWN = 'test_format_md'

TARGETS = TargetBuilder(BuildUnit.by_name('code_style', 'markdown')) \
    .add_phony_target(FORMAT_MARKDOWN).set_runnables(EnforceMarkdownCodeStyle()) \
    .add_phony_target(TEST_FORMAT_MARKDOWN).set_runnables(CheckMarkdownCodeStyle()) \
    .build()
