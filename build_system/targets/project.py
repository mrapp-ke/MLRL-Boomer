"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides information about the project that is important to the build system.
"""
from dataclasses import replace
from functools import reduce
from os import environ
from pathlib import Path
from typing import List, Set

from core.build_unit import BuildUnit
from util.env import get_env_bool
from util.files import FileSearch, FileType
from util.requirements import RequirementVersion

from targets.version_files import DevelopmentVersionFile, PythonVersionFile, SemanticVersion, VersionFile, \
    VersionTextFile


class Project:
    """
    Provides information about the project in general.

    Attributes:
        root_directory: The path to the project's root directory
    """

    root_directory = Path('.')

    @staticmethod
    def version_file() -> VersionFile:
        """
        Returns the file that stores the current version of the project.

        :return: The file that stores the current version of the project
        """
        return VersionFile(Project.BuildSystem.resource_directory / 'versioning' / 'version')

    @staticmethod
    def development_version_file() -> DevelopmentVersionFile:
        """
        Returns the file that stores the current development version of the project.

        :return: The file that stores the current development version of the project
        """
        return DevelopmentVersionFile(Project.BuildSystem.resource_directory / 'versioning' / 'version-dev')

    @staticmethod
    def version(release: bool = False) -> SemanticVersion:
        """
        Returns the current version of the project.

        :param release: True, if the release version should be returned, False, if the development version should be
                        returned
        :return:        The current version of the project
        """
        version = Project.version_file().version

        if release or get_env_bool(environ, 'RELEASE'):
            return version

        return replace(version, dev=Project.development_version())

    @staticmethod
    def development_version() -> int:
        """
        Returns the current development version of the project.

        :return: The current development version of the project
        """
        return Project.development_version_file().development_version

    class BuildSystem:
        """
        Provides information about the project's build system.

        Attributes:
            root_directory:         The path to the build system's root directory
            build_directory_name:   The name of the build system's build directory
        """

        root_directory = BuildUnit.BUILD_SYSTEM_DIRECTORY

        resource_directory = root_directory / 'res'

        build_directory_name = BuildUnit.BUILD_DIRECTORY_NAME

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searching for files within the build system.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.BuildSystem.build_directory_name)

    class Python:
        """
        Provides information about the project's Python code.

        Attributes:
            root_directory:                     The path to the Python code's root directory
            build_directory_name:               The name of the Python code's build directory
            wheel_directory_name:               The name of the directory that contains wheel packages
            wheel_metadata_directory_suffix:    The suffix of the directory that contains the metadata of wheel packages
        """

        root_directory = Path('python')

        build_directory_name = 'build'

        wheel_directory_name = 'dist'

        wheel_metadata_directory_suffix = '.egg-info'

        @staticmethod
        def python_version_file() -> PythonVersionFile:
            """
            Returns the file that stores the Python versions that are supported by the project.

            :return: The file that stores the Python versions
            """
            return PythonVersionFile(Project.Python.root_directory / '.version-python')

        @staticmethod
        def supported_python_versions() -> RequirementVersion:
            """
            Returns the Python versions supported by the project.

            :return: The Python versions supported by the project
            """
            return Project.Python.python_version_file().supported_versions

        @staticmethod
        def minimum_python_version() -> str:
            """
            Returns the minimum Python version required by the project.

            :return: The minimum Python version required by the project
            """
            return RequirementVersion.PREFIX_GEQ + ' ' + Project.Python.supported_python_versions().min_version

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searching for files within the Python code.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.Python.build_directory_name) \
                .exclude_subdirectories_by_name(Project.Python.wheel_directory_name) \
                .exclude_subdirectories_by_name('__pycache__') \
                .exclude_subdirectories_by_substrings(ends_with=Project.Python.wheel_metadata_directory_suffix)

        @staticmethod
        def find_subprojects() -> Set[Path]:
            """
            Finds and returns the paths to all Python subprojects.

            :return: A set that contains the paths to all subprojects that have been found
            """
            return {
                toml_file.parent
                for toml_file in Project.Python.file_search().filter_by_name('pyproject.template.toml').list(
                    Project.Python.root_directory)
            }

        @staticmethod
        def find_test_directories() -> Set[Path]:
            """
            Finds and returns the names of all directories that contain test files.

            :return: A set that contains the names of all directories that have been found
            """
            test_files: List[Path] = []
            test_files = reduce(
                lambda aggr, suffix: aggr + Project.Python.file_search() \
                    .filter_by_substrings(starts_with='test_', ends_with='.' + suffix) \
                    .list(Project.Python.root_directory / 'tests'),
                FileType.python().suffixes,
                test_files)
            return {test_file.parent for test_file in test_files}

    class Cpp:
        """
        Provides information about the project's C++ code.

        Attributes:
            root_directory:         The path to the C++ code's root directory
            build_directory_name:   The name of the C++ code's build directory
        """

        root_directory = Path('cpp')

        build_directory_name = 'build'

        @staticmethod
        def cpp_version() -> str:
            """
            Returns the C++ version that should be used for compilation.

            :return: The C++ version that should be used for compilation
            """
            return VersionTextFile(Project.Cpp.root_directory / '.version-cpp').version_string

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searchin for files within the C++ code.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.Cpp.build_directory_name) \
                .exclude_subdirectories_by_name('xsimd')

        @staticmethod
        def find_subprojects() -> Set[Path]:
            """
            Finds and returns the paths to all C++ subprojects.

            :return: A set that contains the paths to all subprojects that have been found
            """
            return {
                meson_file.parent
                for meson_file in Project.Cpp.file_search() \
                    .exclude_subdirectories_by_name('packagefiles') \
                    .filter_by_name('meson.build') \
                    .list(Project.Cpp.root_directory) if not meson_file.parent.samefile(Project.Cpp.root_directory)
            }

    class Documentation:
        """
        Provides information about the project's documentation.

        Attributes:
            root_directory: The path to the documentation's root directory
        """

        root_directory = Path('doc')

        apidoc_directory = root_directory / 'developer_guide' / 'api'

        build_directory_name = '_build'

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searching for files within the documentation.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.Documentation.build_directory_name)

    class Github:
        """
        Provides information about the project's GitHub-related files.

        Attributes:
            root_directory: The path to the root directory that contains all GitHub-related files
        """

        root_directory = Path('.github')

    class Readthedocs:
        """
        Provides information about the project's readthedocs-related files.

        Attributes:
            root_directory: The path to the root directory that contains all readthedocs-related files
        """

        root_directory = Path('.readthedocs')
