"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python package.
"""
from dataclasses import dataclass, field
from typing import List

import pkg_resources

from mlrl.common.cython.info import CppLibraryInfo, get_cpp_library_info


@dataclass
class PythonPackageInfo:
    """
    Provides information about a Python package.

    Attributes:
        package_name:       A string that specifies the package name
        package_version:    A string that specifies the package version
        python_packages:    A list that contains a `PythonPackageInfo` for each Python package used by this package
        cpp_libraries:      A list that contains a `CppLibraryInfo` for each C++ library used by this package
    """
    package_name: str
    python_packages: List['PythonPackageInfo'] = field(default_factory=list)
    cpp_libraries: List[CppLibraryInfo] = field(default_factory=list)

    @property
    def package_version(self) -> str:
        """
        The version of the Python package.
        """
        return pkg_resources.get_distribution(self.package_name).version

    def __str__(self) -> str:
        return self.package_name + ' ' + self.package_version


def get_package_info() -> PythonPackageInfo:
    """
    Returns information about this Python package.

    :return: A `PythonPackageInfo` that provides information about the Python package
    """
    return PythonPackageInfo(package_name='mlrl-common', cpp_libraries=[get_cpp_library_info()])
