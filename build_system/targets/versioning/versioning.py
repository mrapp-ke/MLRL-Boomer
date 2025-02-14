"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for updating the project's version.
"""
from dataclasses import replace

from core.build_unit import BuildUnit
from util.log import Log

from targets.version_files import DevelopmentVersionFile, VersionFile


def __get_version_file() -> VersionFile:
    version_file = VersionFile()
    Log.info('Current version is "%s"', str(version_file.version))
    return version_file


def __get_development_version_file() -> DevelopmentVersionFile:
    version_file = DevelopmentVersionFile()
    Log.info('Current development version is "%s"', str(version_file.development_version))
    return version_file


def increment_development_version(_: BuildUnit):
    """
    Increments the development version.
    """
    version_file = __get_development_version_file()
    version_file.update(version_file.development_version + 1)


def reset_development_version(_: BuildUnit):
    """
    Resets the development version.
    """
    version_file = __get_development_version_file()
    version_file.update(0)


def increment_patch_version(_: BuildUnit):
    """
    Increments the patch version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, patch=version.patch + 1))


def increment_minor_version(_: BuildUnit):
    """
    Increments the minor version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, minor=version.minor + 1, patch=0))


def increment_major_version(_: BuildUnit):
    """
    Increments the major version.
    """
    version_file = __get_version_file()
    version = version_file.version
    version_file.update(replace(version, major=version.major + 1, minor=0, patch=0))
