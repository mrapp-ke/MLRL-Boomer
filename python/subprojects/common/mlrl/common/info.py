"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python package.
"""
from typing import List

import pkg_resources

from mlrl.common.cython.info import CppLibraryInfo, get_common_cpp_library_info


class PythonPackageInfo:
    """
    Provides information about a Python package.
    """

    def __init__(self,
                 package_name: str,
                 python_packages: List['PythonPackageInfo'] = [],
                 cpp_libraries: List[CppLibraryInfo] = []):
        """
        :param package_name:    A string that specifies the package name
        :param package_version: A string that specifies the package version
        :param python_packages: A list that contains a `PythonPackageInfo` for each Python package used by this package
        :param cpp_libraries:   A list that contains a `CppLibraryInfo` for each C++ library used by this package
        """
        self.package_name = package_name
        self.package_version = pkg_resources.get_distribution(package_name).version
        self.python_packages = python_packages
        self.cpp_libraries = cpp_libraries

    def get_package_name(self) -> str:
        """
        Returns the name of the Python package.

        :return: A string that specifies the package name
        """
        return self.package_name

    def get_package_version(self) -> str:
        """
        Returns the version of the Python package.

        :return A string that specifies the package version
        """
        return self.package_version

    def get_python_packages(self) -> List['PythonPackageInfo']:
        """
        Returns information about all Python packages that are used by this Python packages.

        :return: A list that contains a `PythonPackageInfo` for each Python package used by this package
        """
        return self.python_packages

    def get_cpp_libraries(self) -> List[CppLibraryInfo]:
        """
        Returns information about all C++ libraries that are used by this Python package.

        :return: A list that contains a `CppLibraryInfo` for each C++ library used by this package
        """
        return self.cpp_libraries

    def __str__(self) -> str:
        return self.get_package_name() + ' ' + self.get_package_version()


def get_package_info() -> PythonPackageInfo:
    """
    Returns information about this Python package.

    :return: A `PythonPackageInfo` that provides information about the Python package
    """
    return PythonPackageInfo(package_name='mlrl-common', cpp_libraries=[get_common_cpp_library_info()])
