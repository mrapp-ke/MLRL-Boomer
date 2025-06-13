"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python package.
"""
from dataclasses import dataclass, field
from importlib.metadata import requires, version
from typing import Set

from packaging.requirements import Requirement

from mlrl.common.cython.package_info import CppLibraryInfo, get_cpp_library_info


@dataclass
class RuleLearnerPackageInfo:
    """
    Provides information about a Python package that implements a rule learner.

    Attributes:
        package_name:       A string that specifies the package name
        python_packages:    A set that contains a `RuleLearnerPackageInfo` for each Python package used by this package
        cpp_libraries:      A set that contains a `CppLibraryInfo` for each C++ library used by this package
    """
    package_name: str
    python_packages: Set['RuleLearnerPackageInfo'] = field(default_factory=set)
    cpp_libraries: Set[CppLibraryInfo] = field(default_factory=set)

    @property
    def package_version(self) -> str:
        """
        The version of the Python package.
        """
        return version(self.package_name)

    @property
    def dependencies(self) -> Set['RuleLearnerPackageInfo']:
        """
        A set that contains a `RuleLearnerPackageInfo` for each dependency of this package.
        """
        dependencies = requires(self.package_name)
        package_infos = {RuleLearnerPackageInfo(package_name=Requirement(dependency).name) for dependency in dependencies}

        for python_package in self.python_packages:
            package_infos.discard(python_package)

        return package_infos

    def __str__(self) -> str:
        return self.package_name + ' ' + self.package_version

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


def get_package_info() -> RuleLearnerPackageInfo:
    """
    Returns information about this Python package.

    :return: A `RuleLearnerPackageInfo` that provides information about the Python package
    """
    return RuleLearnerPackageInfo(package_name='mlrl-common', cpp_libraries={get_cpp_library_info()})
