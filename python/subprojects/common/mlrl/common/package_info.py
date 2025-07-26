"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python packages that implement rule learners.
"""
from dataclasses import dataclass, field
from importlib.metadata import requires, version
from typing import Any, Set, override

from packaging.requirements import Requirement

from mlrl.common.cython.package_info import CppLibraryInfo, get_cpp_library_info


@dataclass
class PackageInfo:
    """
    Provides information about a Python package that implements a rule learner.

    Attributes:
        package_name:       A string that specifies the package name
        python_packages:    A set that contains a `PackageInfo` for each Python package used by this package
        cpp_libraries:      A set that contains a `CppLibraryInfo` for each C++ library used by this package
    """
    package_name: str
    python_packages: Set['PackageInfo'] = field(default_factory=set)
    cpp_libraries: Set[CppLibraryInfo] = field(default_factory=set)

    @property
    def package_version(self) -> str:
        """
        The version of the Python package.
        """
        return version(self.package_name)

    @property
    def dependencies(self) -> Set['PackageInfo']:
        """
        A set that contains a `PackageInfo` for each dependency of this package.
        """
        package_infos: Set[PackageInfo] = set()
        dependencies = requires(self.package_name)

        if dependencies:
            for dependency in dependencies:
                package_infos.add(PackageInfo(package_name=Requirement(dependency).name))

        for python_package in self.python_packages:
            package_infos.discard(python_package)

        return package_infos

    @override
    def __str__(self) -> str:
        return self.package_name + ' ' + self.package_version

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) \
            and self.package_name == other.package_name \
            and self.package_version == other.package_version

    @override
    def __hash__(self) -> int:
        return hash((self.package_name, self.package_version))


def get_package_info() -> PackageInfo:
    """
    Returns information about this Python package.

    :return: A `PackageInfo` that provides information about the Python package
    """
    return PackageInfo(package_name='mlrl-common', cpp_libraries={get_cpp_library_info()})
