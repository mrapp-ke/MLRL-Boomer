"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for Markdown files.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileSearch, FileType

from targets.code_style.markdown.targets import CheckMarkdownCodeStyle, EnforceMarkdownCodeStyle
from targets.code_style.modules import CodeModule
from targets.paths import Project

FORMAT_MARKDOWN = 'format_md'

TEST_FORMAT_MARKDOWN = 'test_format_md'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(FORMAT_MARKDOWN).set_runnables(EnforceMarkdownCodeStyle()) \
    .add_phony_target(TEST_FORMAT_MARKDOWN).set_runnables(CheckMarkdownCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.markdown(),
        root_directory=Project.root_directory,
        source_file_search=FileSearch().set_recursive(False),
    ),
    CodeModule(
        file_type=FileType.markdown(),
        root_directory=Project.Python.root_directory,
        source_file_search=Project.Python.file_search(),
    ),
    CodeModule(
        file_type=FileType.markdown(),
        root_directory=Project.Documentation.root_directory,
        source_file_search=Project.Documentation.file_search(),
    ),
]
