"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides access to directories and files belonging to different modules that are part of the project.
"""
from abc import ABC, abstractmethod
from os import path


class Module(ABC):
    """
    An abstract base class for all classes that provide access to directories and files that belong to a module.
    """

    @property
    @abstractmethod
    def root_dir(self) -> str:
        """
        The path to the module's root directory.
        """

    @property
    def build_dir(self) -> str:
        """
        The path to the directory, where build files should be stored.
        """
        return path.join(self.root_dir, 'build')

    @property
    def requirements_file(self) -> str:
        """
        The path to the requirements.txt file that specifies dependencies required by a module.
        """
        return path.join(self.root_dir, 'requirements.txt')


class PythonModule(Module):
    """
    Provides access to directories and files that belong to the project's Python code.
    """

    @property
    def root_dir(self) -> str:
        return 'python'


class CppModule(Module):
    """
    Provides access to directories and files that belong to the project's C++ code.
    """

    @property
    def root_dir(self) -> str:
        return 'cpp'


class BuildModule(Module):
    """
    Provides access to directories and files that belong to the build system.
    """

    @property
    def root_dir(self) -> str:
        return 'scons'

    @property
    def build_dir(self) -> str:
        return 'build'


BUILD_MODULE = BuildModule()

PYTHON_MODULE = PythonModule()

CPP_MODULE = CppModule()
