"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for updating the project's version.
"""
import sys

from dataclasses import dataclass
from typing import Optional

VERSION_FILE = '.version'

DEV_VERSION_FILE = '.version-dev'

VERSION_FILE_ENCODING = 'utf-8'


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


def __read_version_file(version_file) -> str:
    with open(version_file, mode='r', encoding=VERSION_FILE_ENCODING) as file:
        lines = file.readlines()

        if len(lines) != 1:
            print('File "' + version_file + '" must contain exactly one line')
            sys.exit(-1)

        return lines[0]


def __write_version_file(version_file, version: str):
    with open(version_file, mode='w', encoding=VERSION_FILE_ENCODING) as file:
        file.write(version)


def __get_current_development_version() -> int:
    current_version = __read_version_file(DEV_VERSION_FILE)
    print('Current development version is "' + current_version + '"')
    return Version.parse_version_number(current_version)


def __update_development_version(dev: int):
    updated_version = str(dev)
    print('Updated version to "' + updated_version + '"')
    __write_version_file(DEV_VERSION_FILE, updated_version)


def __get_current_version() -> Version:
    current_version = __read_version_file(VERSION_FILE)
    print('Current version is "' + current_version + '"')
    return Version.parse(current_version)


def __update_version(version: Version):
    updated_version = str(version)
    print('Updated version to "' + updated_version + '"')
    __write_version_file(VERSION_FILE, updated_version)


def get_current_version() -> Version:
    """
    Returns the project's current version.

    :return: The project's current version
    """
    return Version.parse(__read_version_file(VERSION_FILE))


def print_current_version():
    """
    Prints the project's current version.
    """
    return print(str(get_current_version()))


def increment_development_version():
    """
    Increments the development version.
    """
    dev = __get_current_development_version()
    dev += 1
    __update_development_version(dev)


def reset_development_version():
    """
    Resets the development version.
    """
    __get_current_development_version()
    __update_development_version(0)


def apply_development_version():
    """
    Appends the development version to the current semantic version.
    """
    version = __get_current_version()
    version.dev = __get_current_development_version()
    __update_version(version)


def increment_patch_version():
    """
    Increments the patch version.
    """
    version = __get_current_version()
    version.patch += 1
    __update_version(version)


def increment_minor_version():
    """
    Increments the minor version.
    """
    version = __get_current_version()
    version.minor += 1
    version.patch = 0
    __update_version(version)


def increment_major_version():
    """
    Increments the major version.
    """
    version = __get_current_version()
    version.major += 1
    version.minor = 0
    version.patch = 0
    __update_version(version)
