"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets and modules for installing C++ dependencies that are required by the project.
"""

from targets.dependencies.cpp.modules import WrapFileModule
from targets.project import Project

MODULES = [
    WrapFileModule(
        root_directory=Project.Cpp.root_directory,
        wrap_file_search=Project.Cpp.file_search(),
    ),
]
