"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for Python and Cython files.
"""
from code_style.modules import CodeModule
from code_style.python.targets import CheckCythonCodeStyle, CheckPythonCodeStyle, EnforceCythonCodeStyle, \
    EnforcePythonCodeStyle
from util.files import FileSearch
from util.languages import Language
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_PYTHON = 'format_python'

TEST_FORMAT_PYTHON = 'test_format_python'

TARGETS = TargetBuilder(BuildUnit('code_style', 'python')) \
    .add_phony_target(FORMAT_PYTHON).set_runnables(EnforcePythonCodeStyle(), EnforceCythonCodeStyle()) \
    .add_phony_target(TEST_FORMAT_PYTHON).set_runnables(CheckPythonCodeStyle(), CheckCythonCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        language=Language.PYTHON,
        root_directory='scons',
    ),
    CodeModule(
        language=Language.PYTHON,
        root_directory='python',
        source_file_search=FileSearch() \
            .set_recursive(True) \
            .exclude_subdirectories_by_name('build', 'dist', '__pycache__') \
            .exclude_subdirectories_by_substrings(ends_with='.egg.info'),
    ),
    CodeModule(
        language=Language.CYTHON,
        root_directory='python',
        source_file_search=FileSearch() \
            .set_recursive(True) \
            .exclude_subdirectories_by_name('build', 'dist', '__pycache__') \
            .exclude_subdirectories_by_substrings(ends_with='.egg.info'),
    ),
    CodeModule(
        language=Language.PYTHON,
        root_directory='doc',
    ),
]
