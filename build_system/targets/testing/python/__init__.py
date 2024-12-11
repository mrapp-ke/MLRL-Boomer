"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for testing Python code.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder

from targets.packaging import INSTALL_WHEELS
from targets.paths import Project
from targets.testing.python.modules import PythonTestModule
from targets.testing.python.targets import TestPython

TESTS_PYTHON = 'tests_python'

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target(TESTS_PYTHON) \
        .depends_on(INSTALL_WHEELS) \
        .set_runnables(TestPython()) \
    .build()

MODULES = [
    PythonTestModule(
        root_directory=Project.Python.root_directory,
        build_directory_name=Project.Python.build_directory_name,
        test_file_search=Project.Python.file_search() \
            .filter_subdirectories_by_name(Project.Python.test_directory_name),
    ),
]
