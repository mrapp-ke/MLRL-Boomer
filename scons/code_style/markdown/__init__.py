"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for Markdown files.
"""
from code_style.markdown.targets import CheckMarkdownCodeStyle, EnforceMarkdownCodeStyle
from code_style.modules import CodeModule
from util.files import FileSearch
from util.languages import Language
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_MARKDOWN = 'format_md'

TEST_FORMAT_MARKDOWN = 'test_format_md'

TARGETS = TargetBuilder(BuildUnit('code_style', 'markdown')) \
    .add_phony_target(FORMAT_MARKDOWN).set_runnables(EnforceMarkdownCodeStyle()) \
    .add_phony_target(TEST_FORMAT_MARKDOWN).set_runnables(CheckMarkdownCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='.',
        source_file_search=FileSearch().set_recursive(False),
    ),
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='python',
    ),
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='doc',
    ),
]
