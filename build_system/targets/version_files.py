"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing version files.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from os import environ
from typing import Optional

from util.env import get_env, get_env_bool
from util.io import TextFile
from util.log import Log


@dataclass
class Version:
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
    def parse_version_number(version_number: str) -> int:
        """
        Parses and returns a single version number from a given string.

        :param version_number:  The string to be parsed
        :return:                The version number that has been parsed
        """
        try:
            number = int(version_number)

            if number < 0:
                raise ValueError()

            return number
        except ValueError as error:
            raise ValueError('Version numbers must be non-negative integers, but got: ' + version_number) from error

    @staticmethod
    def parse(version: str) -> 'Version':
        """
        Parses and returns a version from a given string.

        :param version: The string to be parsed
        :return:        The version that has been parsed
        """
        parts = version.split('.')

        if len(parts) != 3:
            raise ValueError('Version must be given in format MAJOR.MINOR.PATCH, but got: ' + version)

        major = Version.parse_version_number(parts[0])
        minor = Version.parse_version_number(parts[1])
        patch = Version.parse_version_number(parts[2])
        return Version(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        version = str(self.major) + '.' + str(self.minor) + '.' + str(self.patch)

        if self.dev is not None:
            version += '.dev' + str(self.dev)

        return version


class VersionFile(TextFile):
    """
    The file that stores the project's version.
    """

    def __init__(self):
        super().__init__('.version')

    @cached_property
    def version(self) -> Version:
        """
        The version that is stored in the file.
        """
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + self.file + '" must contain exactly one line')

        return Version.parse(lines[0])

    def update(self, version: Version):
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


class DevelopmentVersionFile(TextFile):
    """
    The file that stores the project's development version.
    """

    def __init__(self):
        super().__init__('.version-dev')

    @cached_property
    def development_version(self) -> int:
        """
        The development version that is stored in the file.
        """
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + self.file + '" must contain exactly one line')

        return Version.parse_version_number(lines[0])

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


def get_project_version(release: bool = False) -> Version:
    """
    Returns the current version of the project.

    :param release: True, if the release version should be returned, False, if the development version should be
                    returned
    :return:        The current version of the project
    """
    version = VersionFile().version

    if release or (get_env_bool(environ, 'READTHEDOCS') and get_env(environ, 'READTHEDOCS_VERSION_TYPE') == 'tag'):
        return version

    return replace(version, dev=DevelopmentVersionFile().development_version)
