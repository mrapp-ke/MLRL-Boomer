"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for Cython files.
"""

from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder
from targets.code_style.cython.targets import (
    CheckCythonCodeStyle,
    EnforceCythonCodeStyle,
)
from targets.code_style.modules import CodeModule
from targets.project import Project
from util.files import FileType

FORMAT_CYTHON = 'format_cython'

TEST_FORMAT_CYTHON = 'test_format_cython'

TARGETS = (
    TargetBuilder(BuildUnit.for_file(Path(__file__)))
    .add_phony_target(FORMAT_CYTHON)
    .set_runnables(EnforceCythonCodeStyle())
    .add_phony_target(TEST_FORMAT_CYTHON)
    .set_runnables(CheckCythonCodeStyle())
    .build()
)

MODULES = [
    CodeModule(
        file_type=FileType.cython(),
        root_directory=Project.Python.root_directory,
        source_file_search=Project.Python.file_search(),
    ),
]
