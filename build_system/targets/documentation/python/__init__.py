"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for generating API documentations for C++ code.
"""
from os import path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.documentation.python.modules import PythonApidocModule
from targets.documentation.python.targets import ApidocIndexPython, ApidocPython
from targets.packaging import INSTALL_WHEELS
from targets.paths import Project

APIDOC_PYTHON = 'apidoc_python'

APIDOC_PYTHON_INDEX = 'apidoc_python_index'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_build_target(APIDOC_PYTHON) \
        .depends_on(INSTALL_WHEELS) \
        .set_runnables(ApidocPython()) \
    .add_build_target(APIDOC_PYTHON_INDEX) \
        .depends_on(APIDOC_PYTHON) \
        .set_runnables(ApidocIndexPython()) \
    .build()

MODULES = [
    PythonApidocModule(root_directory=subproject,
                       output_directory=path.join(Project.Documentation.apidoc_directory, 'python',
                                                  path.basename(subproject)),
                       source_directory_name='mlrl',
                       source_file_search=Project.Python.file_search())
    for subproject in Project.Python.find_subprojects()
]
