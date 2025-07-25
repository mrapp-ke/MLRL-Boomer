"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for testing Python code.
"""
from functools import reduce
from os import path

from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from util.files import FileType

from targets.packaging import INSTALL_WHEELS
from targets.project import Project
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
        root_directory=test_directory,
        result_directory=path.join(Project.Python.root_directory, 'tests', Project.Python.build_directory_name,
                                   'test-results'),
    ) for test_directory in Project.Python.find_test_directories()
]
