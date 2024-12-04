"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for C++ files.
"""
from code_style.cpp.targets import CheckCppCodeStyle, EnforceCppCodeStyle
from code_style.modules import CodeModule
from util.languages import Language
from util.paths import Project
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

FORMAT_CPP = 'format_cpp'

TEST_FORMAT_CPP = 'test_format_cpp'

TARGETS = TargetBuilder(BuildUnit('code_style', 'cpp')) \
    .add_phony_target(FORMAT_CPP).set_runnables(EnforceCppCodeStyle()) \
    .add_phony_target(TEST_FORMAT_CPP).set_runnables(CheckCppCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        language=Language.CPP,
        root_directory=Project.Cpp.root_directory,
        source_file_search=Project.Cpp.file_search(),
    ),
]
