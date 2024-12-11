"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for C++ files.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileType

from targets.code_style.cpp.targets import CheckCppCodeStyle, EnforceCppCodeStyle
from targets.code_style.modules import CodeModule
from targets.paths import Project

FORMAT_CPP = 'format_cpp'

TEST_FORMAT_CPP = 'test_format_cpp'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(FORMAT_CPP).set_runnables(EnforceCppCodeStyle()) \
    .add_phony_target(TEST_FORMAT_CPP).set_runnables(CheckCppCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.cpp(),
        root_directory=Project.Cpp.root_directory,
        source_file_search=Project.Cpp.file_search(),
    ),
]
