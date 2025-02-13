"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for generating documentations.
"""
from os import path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.documentation.cpp import APIDOC_CPP, APIDOC_CPP_INDEX
from targets.documentation.modules import SphinxModule
from targets.documentation.python import APIDOC_PYTHON, APIDOC_PYTHON_INDEX
from targets.documentation.targets import BuildDocumentation
from targets.project import Project

APIDOC_INDEX = 'apidoc_index'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target('apidoc') \
        .depends_on(APIDOC_CPP, APIDOC_PYTHON, clean_dependencies=True) \
        .nop() \
    .add_phony_target(APIDOC_INDEX) \
        .depends_on(APIDOC_CPP_INDEX, APIDOC_PYTHON_INDEX, clean_dependencies=True) \
        .nop() \
    .add_build_target('doc') \
        .depends_on(APIDOC_INDEX, clean_dependencies=True) \
        .set_runnables(BuildDocumentation()) \
    .build()

MODULES = [
    SphinxModule(
        root_directory=Project.Documentation.root_directory,
        output_directory=path.join(Project.Documentation.root_directory, Project.Documentation.build_directory_name),
        source_file_search=Project.Documentation.file_search(),
    ),
]
