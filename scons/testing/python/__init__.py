"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for testing Python code.
"""
from packaging import INSTALL_WHEELS
from testing.python.modules import PythonTestModule
from testing.python.targets import TestPython
from util.paths import Project
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

TESTS_PYTHON = 'tests_python'

TARGETS = TargetBuilder(BuildUnit('testing', 'python')) \
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
