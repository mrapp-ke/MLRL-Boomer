"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python packages that implement rule learners.
"""
from dataclasses import dataclass, field
from importlib.metadata import requires
from typing import Set

from packaging.requirements import Requirement

from mlrl.common.cython.package_info import CppLibraryInfo, get_cpp_library_info

from mlrl.util.package_info import PackageInfo


@dataclass
class RuleLearnerPackageInfo:
    """
    Provides information about a Python package that implements a rule learner.

    Attributes:
        package_info:       Information about the Python package
        cpp_libraries:      A set that contains a `CppLibraryInfo` for each C++ library used by this package
    """
    package_info: PackageInfo
    cpp_libraries: Set[CppLibraryInfo] = field(default_factory=set)

    @property
    def dependencies(self) -> Set['PackageInfo']:
        """
        A set that contains a `PackageInfo` for each dependency of this package.
        """
        package_info = self.package_info
        dependencies = requires(package_info.package_name)
        package_infos = {PackageInfo(package_name=Requirement(dependency).name) for dependency in dependencies}

        for python_package in package_info.python_packages:
            package_infos.discard(python_package)

        return package_infos

    def __str__(self) -> str:
        return str(self.package_info)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.package_info == other.package_info

    def __hash__(self):
        return hash(self.package_info)


def get_package_info() -> RuleLearnerPackageInfo:
    """
    Returns information about this Python package.

    :return: A `RuleLearnerPackageInfo` that provides information about the Python package
    """
    return RuleLearnerPackageInfo(package_info=PackageInfo(package_name='mlrl-common'),
                                  cpp_libraries={get_cpp_library_info()})
