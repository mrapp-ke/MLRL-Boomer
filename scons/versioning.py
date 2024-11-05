"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for updating the project's version.
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


def __parse_version_number(version_number: str) -> int:
    try:
        number = int(version_number)

        if number < 0:
            raise ValueError()

        return number
    except ValueError:
        print('Version numbers must only consist of non-negative integers, but got: ' + version_number)
        sys.exit(-1)


def __get_current_development_version() -> int:
    current_version = __read_version_file(DEV_VERSION_FILE)
    print('Current development version is "' + current_version + '"')
    return __parse_version_number(current_version)


def __update_development_version(dev: int):
    updated_version = str(dev)
    print('Updated version to "' + updated_version + '"')
    __write_version_file(DEV_VERSION_FILE, updated_version)


def __parse_version(version: str) -> Version:
    parts = version.split('.')

    if len(parts) != 3:
        print('Version must be given in format MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH.devN, but got: ' + version)
        sys.exit(-1)

    major = __parse_version_number(parts[0])
    minor = __parse_version_number(parts[1])
    patch = __parse_version_number(parts[2])
    return Version(major=major, minor=minor, patch=patch)


def __get_current_version() -> Version:
    current_version = __read_version_file(VERSION_FILE)
    print('Current version is "' + current_version + '"')
    return __parse_version(current_version)


def __update_version(version: Version):
    updated_version = str(version)
    print('Updated version to "' + updated_version + '"')
    __write_version_file(VERSION_FILE, updated_version)


def increment_development_version(**_):
    """
    Increments the development version.
    """
    dev = __get_current_development_version()
    dev += 1
    __update_development_version(dev)


def reset_development_version(**_):
    """
    Resets the development version.
    """
    __get_current_development_version()
    __update_development_version(0)


def apply_development_version(**_):
    """
    Appends the development version to the current semantic version.
    """
    version = __get_current_version()
    version.dev = __get_current_development_version()
    __update_version(version)


def increment_patch_version(**_):
    """
    Increments the patch version.
    """
    version = __get_current_version()
    version.patch += 1
    __update_version(version)


def increment_minor_version(**_):
    """
    Increments the minor version.
    """
    version = __get_current_version()
    version.minor += 1
    version.patch = 0
    __update_version(version)


def increment_major_version(**_):
    """
    Increments the major version.
    """
    version = __get_current_version()
    version.major += 1
    version.minor = 0
    version.patch = 0
    __update_version(version)
