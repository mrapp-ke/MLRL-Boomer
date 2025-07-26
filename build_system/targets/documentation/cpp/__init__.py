"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for generating API documentations for C++ code.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.documentation.cpp.modules import CppApidocModule
from targets.documentation.cpp.targets import ApidocCpp, ApidocIndexCpp, UpdateDoxyfile
from targets.project import Project

APIDOC_CPP = 'apidoc_cpp'

APIDOC_CPP_INDEX = 'apidoc_cpp_index'

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_build_target(APIDOC_CPP) \
        .set_runnables(ApidocCpp()) \
    .add_build_target(APIDOC_CPP_INDEX) \
        .depends_on(APIDOC_CPP, clean_dependencies=True) \
        .set_runnables(ApidocIndexCpp()) \
    .add_phony_target('update_doxyfile') \
        .set_runnables(UpdateDoxyfile()) \
    .build()

MODULES = [
    CppApidocModule(
        root_directory=subproject,
        output_directory=Project.Documentation.apidoc_directory / 'cpp' / subproject.name,
        include_directory_name='include',
    ) for subproject in Project.Cpp.find_subprojects() if (subproject / 'include').is_dir()
]
