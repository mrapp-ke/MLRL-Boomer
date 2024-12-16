"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for generating API documentations for C++ code.
"""
from os import path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.documentation.cpp.modules import CppApidocModule
from targets.documentation.cpp.targets import ApidocCpp, ApidocIndexCpp, UpdateDoxyfile
from targets.paths import Project

APIDOC_CPP = 'apidoc_cpp'

APIDOC_CPP_INDEX = 'apidoc_cpp_index'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_build_target(APIDOC_CPP) \
        .set_runnables(ApidocCpp()) \
    .add_build_target(APIDOC_CPP_INDEX) \
        .depends_on(APIDOC_CPP) \
        .set_runnables(ApidocIndexCpp()) \
    .add_phony_target('update_doxyfile') \
        .set_runnables(UpdateDoxyfile()) \
    .build()

MODULES = [
    CppApidocModule(
        root_directory=path.dirname(meson_file),
        output_directory=path.join(Project.Documentation.apidoc_directory, 'cpp',
                                   path.basename(path.dirname(meson_file))),
        project_name=path.basename(path.dirname(meson_file)),
        include_directory_name='include',
    ) for meson_file in Project.Cpp.file_search().filter_by_name('meson.build').list(Project.Cpp.root_directory)
    if path.isdir(path.join(path.dirname(meson_file), 'include'))
]
