"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for updating the project's version.
"""
import sys

from typing import Optional, Tuple

VERSION_FILE = '.version'

DEV_VERSION_FILE = '.version-dev'

VERSION_FILE_ENCODING = 'utf-8'


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


def __parse_version(version: str) -> Tuple[int, int, int]:
    parts = version.split('.')

    if len(parts) != 3:
        print('Version must be given in format MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH.devN, but got: ' + version)
        sys.exit(-1)

    major = __parse_version_number(parts[0])
    minor = __parse_version_number(parts[1])
    patch = __parse_version_number(parts[2])
    return major, minor, patch


def __format_version(major: int, minor: int, patch: int, dev: Optional[int] = None) -> str:
    version = str(major) + '.' + str(minor) + '.' + str(patch)

    if dev is not None:
        version += '.dev' + str(dev)

    return version


def __get_current_version() -> Tuple[int, int, int]:
    current_version = __read_version_file(VERSION_FILE)
    print('Current version is "' + current_version + '"')
    return __parse_version(current_version)


def __update_version(major: int, minor: int, patch: int, dev: Optional[int] = None):
    updated_version = __format_version(major, minor, patch, dev)
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
    major, minor, patch = __get_current_version()
    dev = __get_current_development_version()
    __update_version(major, minor, patch, dev)


def increment_patch_version(**_):
    """
    Increments the patch version.
    """
    major, minor, patch = __get_current_version()
    patch += 1
    __update_version(major, minor, patch)


def increment_minor_version(**_):
    """
    Increments the minor version.
    """
    major, minor, patch = __get_current_version()
    minor += 1
    patch = 0
    __update_version(major, minor, patch)


def increment_major_version(**_):
    """
    Increments the major version.
    """
    major, minor, patch = __get_current_version()
    major += 1
    minor = 0
    patch = 0
    __update_version(major, minor, patch)
