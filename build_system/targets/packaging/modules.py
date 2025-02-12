"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python code that can be built as wheel packages.
"""
from os import environ, path
from typing import Optional

from core.modules import Module
from util.files import FileSearch

from targets.modules import SubprojectModule
from targets.version_files import get_project_version


class PythonPackageModule(SubprojectModule):
    """
    A module that provides access to Python code that can be built as wheel packages.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonPackageModule`.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, PythonPackageModule) and SubprojectModule.Filter.from_env(
                environ, always_match={SubprojectModule.SUBPROJECT_COMMON, SubprojectModule.SUBPROJECT_TESTBED
                                       }).matches(module)

    def __init__(self, root_directory: str, wheel_directory_name: str):
        """
        :param root_directory:          The path to the module's root directory
        :param wheel_directory_name:    The name of the directory that contains wheel packages
        """
        self.root_directory = root_directory
        self.wheel_directory_name = wheel_directory_name

    @property
    def pyproject_toml_file(self) -> str:
        """
        The path to the pyproject.toml file that specifies the meta-data of the package.
        """
        return path.join(self.root_directory, 'pyproject.toml')

    @property
    def wheel_directory(self) -> str:
        """
        Returns the path of the directory that contains the wheel packages that have been built for the module.
        """
        return path.join(self.root_directory, self.wheel_directory_name)

    def find_wheel(self) -> Optional[str]:
        """
        Finds and returns the wheel package that has been built for the module.

        :return: The path to the wheel package or None, if no such package has been found
        """
        wheels = FileSearch() \
            .filter_by_substrings(contains=str(get_project_version(release=True)), ends_with='.whl') \
            .list(self.wheel_directory)
        return wheels[0] if wheels else None

    @property
    def subproject_name(self) -> str:
        return path.basename(self.root_directory)

    def __str__(self) -> str:
        return 'PythonPackageModule {root_directory="' + self.root_directory + '"}'
