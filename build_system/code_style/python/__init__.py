"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for Python and Cython files.
"""
from code_style.modules import CodeModule
from code_style.python.targets import CheckCythonCodeStyle, CheckPythonCodeStyle, EnforceCythonCodeStyle, \
    EnforcePythonCodeStyle
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileType
from util.paths import Project

FORMAT_PYTHON = 'format_python'

TEST_FORMAT_PYTHON = 'test_format_python'

TARGETS = TargetBuilder(BuildUnit('code_style', 'python')) \
    .add_phony_target(FORMAT_PYTHON).set_runnables(EnforcePythonCodeStyle(), EnforceCythonCodeStyle()) \
    .add_phony_target(TEST_FORMAT_PYTHON).set_runnables(CheckPythonCodeStyle(), CheckCythonCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.python(),
        root_directory=Project.BuildSystem.root_directory,
        source_file_search=Project.BuildSystem.file_search(),
    ),
    CodeModule(
        file_type=FileType.python(),
        root_directory=Project.Python.root_directory,
        source_file_search=Project.Python.file_search(),
    ),
    CodeModule(
        file_type=FileType.cython(),
        root_directory=Project.Python.root_directory,
        source_file_search=Project.Python.file_search(),
    ),
    CodeModule(
        file_type=FileType.python(),
        root_directory=Project.Documentation.root_directory,
        source_file_search=Project.Documentation.file_search(),
    ),
]
