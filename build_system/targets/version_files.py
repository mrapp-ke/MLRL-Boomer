"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing version files.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, override

from util.io import TextFile
from util.log import Log
from util.requirements import RequirementVersion
from util.version import Version


@dataclass
class SemanticVersion:
    """
    Represents a semantic version.

    Attributes:
        major:  The major version number
        minor:  The minor version number
        patch:  The patch version number
        dev:    The development version number
    """
    major: int
    minor: int
    patch: int
    dev: Optional[int] = None

    @staticmethod
    def parse(value: str) -> 'SemanticVersion':
        """
        Parses and returns a version from a given string.

        :param value:   The string to be parsed
        :return:        The version that has been parsed
        """
        version = Version.parse(value)

        if len(version.numbers) != 3:
            raise ValueError('Version must be given in format MAJOR.MINOR.PATCH, but got: ' + value)

        return SemanticVersion(major=version.numbers[0], minor=version.numbers[1], patch=version.numbers[2])

    @override
    def __str__(self) -> str:
        version = str(self.major) + '.' + str(self.minor) + '.' + str(self.patch)
        dev = self.dev

        if dev is not None:
            version += '.dev' + str(dev)

        return version


class VersionTextFile(TextFile):
    """
    A text file that stores a version.
    """

    @property
    def version_string(self) -> str:
        """
        The version that is stored in the file as a string.
        """
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + str(self.file) + '" must contain exactly one line')

        return lines[0]


class VersionFile(VersionTextFile):
    """
    The file that stores the project's version.
    """

    @cached_property
    def version(self) -> SemanticVersion:
        """
        The version that is stored in the file.
        """
        return SemanticVersion.parse(self.version_string)

    def update(self, version: SemanticVersion):
        """
        Updates the version that is stored in the file.

        :param version: The version to be stored
        """
        self.write_lines(str(version))
        Log.info('Updated version to "%s"', str(version))

    @override
    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.version
        except AttributeError:
            pass


class DevelopmentVersionFile(VersionTextFile):
    """
    The file that stores the project's development version.
    """

    @cached_property
    def development_version(self) -> int:
        """
        The development version that is stored in the file.
        """
        return Version.parse_version_number(self.version_string)

    def update(self, development_version: int):
        """
        Updates the development version that is stored in the file.

        :param development_version: The development version to be stored
        """
        self.write_lines(str(development_version))
        Log.info('Updated development version to "%s"', str(development_version))

    @override
    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.development_version
        except AttributeError:
            pass


class PythonVersionFile(VersionTextFile):
    """
    The file that stores the supported Python versions.
    """

    @cached_property
    def supported_versions(self) -> RequirementVersion:
        """
        The supported versions that are stored in the file.
        """
        return RequirementVersion.parse(self.version_string)

    def update(self, supported_versions: RequirementVersion):
        """
        Updates the supported Python versions that are stored in the file.

        :param supported_versions: The supported version to be stored
        """
        self.write_lines(str(supported_versions))
        Log.info('Updated supported Python versions to "%s"', str(supported_versions))

    @override
    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.supported_versions
        except AttributeError:
            pass
