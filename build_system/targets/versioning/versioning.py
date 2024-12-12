"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for updating the project's version.
"""
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Optional

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

        if self.dev:
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
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + self.file + '" must contain exactly one line')

        return Version.parse(lines[0])

    def update(self, version: Version):
        self.write_lines(str(version))
        Log.info('Updated version to "%s"', str(version))

    def write_lines(self, *lines: str):
        super().write_lines(lines)

        try:
            del self.version
        except AttributeError:
            pass


class DevelopmentVersionFile(TextFile):
    """
    The file that stores the project's development version.
    """

    @cached_property
    def development_version(self) -> int:
        lines = self.lines

        if len(lines) != 1:
            raise ValueError('File "' + self.file + '" must contain exactly one line')

        return Version.parse_version_number(lines[0])

    def update(self, development_version: int):
        self.write_lines(str(development_version))
        Log.info('Updated development version to "%s"', str(development_version))

    def write_lines(self, *lines: str):
        super().write_lines(lines)

        try:
            del self.development_version
        except AttributeError:
            pass


def __get_version_file() -> VersionFile:
    version_file = VersionFile()
    Log.info('Current version is "%s"', str(version_file.version))
    return version_file


def __get_development_version_file() -> DevelopmentVersionFile:
    version_file = DevelopmentVersionFile()
    Log.info('Current development version is "%s"', str(version_file.development_version))
    return version_file


def get_current_version() -> Version:
    """
    Returns the project's current version.

    :return: The project's current version
    """
    return VersionFile().version


def print_current_version():
    """
    Prints the project's current version.
    """
    return Log.info('%s', str(get_current_version()))


def increment_development_version():
    """
    Increments the development version.
    """
    version_file = __get_development_version_file()
    version_file.update(version_file.dev + 1)


def reset_development_version():
    """
    Resets the development version.
    """
    version_file = __get_development_version_file()
    version_file.update(0)


def apply_development_version():
    """
    Appends the development version to the current semantic version.
    """
    version_file = __get_version_file()
    development_version = __get_development_version_file().development_version
    version_file.update(replace(version_file.version, dev=development_version))


def increment_patch_version():
    """
    Increments the patch version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, patch=version.patch + 1))


def increment_minor_version():
    """
    Increments the minor version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, minor=version.minor + 1, patch=0))


def increment_major_version():
    """
    Increments the major version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, major=version.major + 1, minor=0, patch=0))
