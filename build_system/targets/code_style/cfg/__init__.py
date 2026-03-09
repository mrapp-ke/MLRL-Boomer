"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for checking and enforcing code style definitions for .cfg files.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder
from util.files import FileType

from targets.code_style.cfg.targets import CheckCfgCodeStyle, EnforceCfgCodeStyle
from targets.code_style.modules import CodeModule
from targets.project import Project

FORMAT_CFG = 'format_cfg'

TEST_FORMAT_CFG = 'test_format_cfg'

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target(FORMAT_CFG).set_runnables(EnforceCfgCodeStyle()) \
    .add_phony_target(TEST_FORMAT_CFG).set_runnables(CheckCfgCodeStyle()) \
    .build()

MODULES = [
    CodeModule(
        file_type=FileType.cfg(),
        root_directory=Project.BuildSystem.root_directory,
        source_file_search=Project.BuildSystem.file_search().set_hidden(True),
    ),
    CodeModule(
        file_type=FileType.cfg(),
        root_directory=Project.Cpp.root_directory,
        source_file_search=Project.Cpp.file_search().exclude_by_name('.cpplint.cfg'),
    ),
]
