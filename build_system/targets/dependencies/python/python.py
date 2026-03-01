"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for checking and updating the supported Python versions.
"""
from dataclasses import replace

from core.build_unit import BuildUnit
from util.log import Log
from util.pip import Pip
from util.requirements import RequirementsFiles
from util.version import Version

from targets.project import Project


def __query_latest_python_version(build_unit: BuildUnit) -> Version:
    Pip.install_packages(RequirementsFiles.for_build_unit(build_unit), 'requests')
    # pylint: disable=import-outside-toplevel
    import requests
    url = 'https://raw.githubusercontent.com/actions/python-versions/refs/heads/main/versions-manifest.json'
    Log.verbose('Querying Python versions from ' + url)
    response = requests.get(url, timeout=5)
    available_versions = set()

    for entry in response.json():
        try:
            version_numbers = Version.parse(entry['version']).numbers
            available_versions.add(Version(numbers=(version_numbers[0], version_numbers[1])))
        except ValueError:
            pass

    return sorted(available_versions)[-1]


def check_python_version(build_unit: BuildUnit):
    """
    Checks if the latest Python version is supported.

    :param build_unit: The build unit from which this function is run
    """
    Log.info('Checking if the latest Python version is supported...')
    latest_version = __query_latest_python_version(build_unit)
    version_file = Project.Python.python_version_file()
    supported_version = Version.parse(version_file.supported_versions.max_version)

    if supported_version > latest_version:
        Log.info('Latest Python version %s is supported!', latest_version)
    else:
        version_numbers = supported_version.numbers
        supported_version = Version(numbers=(version_numbers[0], version_numbers[1] - 1))
        Log.info('Latest Python version %s is not supported! The latest supported version is %s.', latest_version,
                 supported_version)


def update_python_version(build_unit: BuildUnit):
    """
    Updates the maximum supported Python version to the latest Python version.

    :param build_unit: The build unit from which this function is run
    """
    latest_version = __query_latest_python_version(build_unit)
    version_file = Project.Python.python_version_file()
    supported_versions = version_file.supported_versions
    supported_version = Version.parse(supported_versions.max_version)

    if supported_version > latest_version:
        Log.info('Latest Python version %s is already supported!', latest_version)
    else:
        version_numbers = latest_version.numbers
        updated_version = Version(numbers=(version_numbers[0], version_numbers[1] + 1))
        version_file.update(replace(supported_versions, max_version=str(updated_version)))
