"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about Python packages.
"""
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Set


@dataclass
class PackageInfo:
    """
    Provides information about a Python package.

    Attributes:
        package_name:       A string that specifies the package name
        python_packages:    A set that contains a `PythonPackageInfo` for each Python package used by this package
    """
    package_name: str
    python_packages: Set['PackageInfo'] = field(default_factory=set)

    @property
    def package_version(self) -> str:
        """
        The version of the Python package.
        """
        return version(self.package_name)

    def __str__(self) -> str:
        return self.package_name + ' ' + self.package_version

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) \
            and self.package_name == other.package_name \
            and self.package_version == other.package_version

    def __hash__(self) -> int:
        return hash((self.package_name, self.package_version))
