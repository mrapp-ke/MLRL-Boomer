"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing version files.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from util.io import TextFile
from util.log import Log
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
    def parse(version: str) -> 'SemanticVersion':
        """
        Parses and returns a version from a given string.

        :param version: The string to be parsed
        :return:        The version that has been parsed
        """
        version = Version.parse(version)

        if len(version.numbers) != 3:
            raise ValueError('Version must be given in format MAJOR.MINOR.PATCH, but got: ' + version)

        return SemanticVersion(major=version.numbers[0], minor=version.numbers[1], patch=version.numbers[2])

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
            raise ValueError('File "' + self.file + '" must contain exactly one line')

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

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.development_version
        except AttributeError:
            pass
