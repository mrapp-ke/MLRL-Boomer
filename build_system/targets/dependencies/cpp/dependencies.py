"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for checking and updating the versions of Python dependencies.
"""

from dataclasses import dataclass
from typing import Any, override

from core.build_unit import BuildUnit
from targets.dependencies.cpp.wrap_file import WrapFile
from util.log import Log
from util.pygithub import GithubApi
from util.version import Version


@dataclass
class Dependency:
    """
    Provides information about an outdated dependency.

    Attributes:
        wrap_file:  The wrap file that declares the dependency
        outdated:   The outdated version of the dependency
        latest:     The latest version of the dependency
    """

    wrap_file: WrapFile
    outdated: Version
    latest: Version

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.wrap_file == other.wrap_file

    @override
    def __hash__(self) -> int:
        return hash(self.wrap_file)


class WrapFileUpdater:
    """
    Allows checking the versions of dependencies declared in Meson wrap files and updating outdated ones.
    """

    def __init__(self, *wrap_files: WrapFile):
        """
        :param wrap_files: The wrap files to be checked
        """
        self.wrap_files = list(wrap_files)

    @staticmethod
    def __query_latest_package_version(build_unit: BuildUnit, wrap_file: WrapFile) -> Version:
        dependency_name = wrap_file.dependency_name
        Log.info(f'Querying latest version of dependency "{dependency_name}"...')
        github_api = GithubApi(build_unit).set_token_from_env()
        tags = github_api.open_repository(wrap_file.repository_name).get_all_tags()
        latest_tag = next(iter(tags), None)

        if not latest_tag:
            raise RuntimeError('No tags available')

        latest_version = Version.parse(latest_tag.name)
        Log.info(f'Latest version of dependency "{dependency_name}" is {latest_version}')
        return latest_version

    def list_outdated_dependencies(self, build_unit: BuildUnit) -> set[Dependency]:
        """
        Returns all outdated dependencies that are declared in the wrap files.

        :param build_unit:  The `BuildUnit` from which this function is invoked
        :return:            A set that contains all outdated dependencies
        """
        outdated_dependencies: set[Dependency] = set()
        version_cache: dict[str, Version] = {}

        for wrap_file in self.wrap_files:
            dependency_name = wrap_file.dependency_name
            latest_version = version_cache.get(dependency_name)

            if not latest_version:
                latest_version = self.__query_latest_package_version(build_unit, wrap_file)
                version_cache[dependency_name] = latest_version

            current_version = wrap_file.version

            if current_version < latest_version:
                outdated_dependencies.add(
                    Dependency(
                        wrap_file=wrap_file,
                        outdated=current_version,
                        latest=latest_version,
                    )
                )

        return outdated_dependencies

    def update_outdated_dependencies(self, build_unit: BuildUnit) -> set[Dependency]:
        """
        Updates all outdated dependencies that are declared in the wrap files.

        :param build_unit:  The `BuildUnit` from which this function is invoked
        :return:            A set that contains all dependencies that have been updated
        """
        outdated_dependencies = self.list_outdated_dependencies(build_unit)

        for outdated_dependency in outdated_dependencies:
            wrap_file = outdated_dependency.wrap_file
            wrap_file.update_version(outdated_dependency.latest)

        return outdated_dependencies
