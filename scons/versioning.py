"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for updating the project's version.
"""
import sys

VERSION_FILE = 'VERSION'

DEV_VERSION_FILE = VERSION_FILE + '.dev'

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


def increment_development_version(**_):
    """
    Increments the development version.
    """
    dev = __get_current_development_version()
    dev += 1
    __update_development_version(dev)
