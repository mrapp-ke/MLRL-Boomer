"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for updating the project's version.
"""
from dataclasses import replace
from functools import cached_property

from util.io import TextFile
from util.log import Log

from targets.version_files import Version, VersionFile


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


def __get_version_file() -> VersionFile:
    version_file = VersionFile()
    Log.info('Current version is "%s"', str(version_file.version))
    return version_file


def __get_development_version_file() -> DevelopmentVersionFile:
    version_file = DevelopmentVersionFile()
    Log.info('Current development version is "%s"', str(version_file.development_version))
    return version_file


def print_current_version():
    """
    Prints the project's current version.
    """
    return Log.info('%s', str(VersionFile().version))


def increment_development_version():
    """
    Increments the development version.
    """
    version_file = __get_development_version_file()
    version_file.update(version_file.development_version + 1)


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
