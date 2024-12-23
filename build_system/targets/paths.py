"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides paths within the project that are important for the build system.
"""
from os import path
from typing import Set

from core.build_unit import BuildUnit
from util.files import FileSearch


class Project:
    """
    Provides paths within the project.

    Attributes:
        root_directory: The path to the project's root directory
    """

    root_directory = '.'

    class BuildSystem:
        """
        Provides paths within the project's build system.

        Attributes:
            root_directory:         The path to the build system's root directory
            build_directory_name:   The name of the build system's build directory
        """

        root_directory = BuildUnit.BUILD_SYSTEM_DIRECTORY

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
        Provides paths within the project's Python code.

        Attributes:
            root_directory:                     The path to the Python code's root directory
            build_directory_name:               The name of the Python code's build directory
            test_directory_name:                The name fo the directory that contains tests
            wheel_directory_name:               The name of the directory that contains wheel packages
            wheel_metadata_directory_suffix:    The suffix of the directory that contains the metadata of wheel packages
        """

        root_directory = 'python'

        build_directory_name = 'build'

        test_directory_name = 'tests'

        wheel_directory_name = 'dist'

        wheel_metadata_directory_suffix = '.egg-info'

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
        def find_subprojects() -> Set[str]:
            """
            Finds and returns the paths to all Python subprojects.

            :return: A set that contains the paths to all subprojects that have been found
            """
            return {
                path.dirname(toml_file)
                for toml_file in Project.Python.file_search().filter_by_name('pyproject.toml').list(
                    Project.Python.root_directory)
            }

    class Cpp:
        """
        Provides paths within the project's C++ code.

        Attributes:
            root_directory:         The path to the C++ code's root directory
            build_directory_name:   The name of the C++ code's build directory
        """

        root_directory = 'cpp'

        build_directory_name = 'build'

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searchin for files within the C++ code.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.Cpp.build_directory_name)

        @staticmethod
        def find_subprojects() -> Set[str]:
            """
            Finds and returns the paths to all C++ subprojects.

            :return: A set that contains the paths to all subprojects that have been found
            """
            return {
                path.dirname(meson_file)
                for meson_file in Project.Cpp.file_search().filter_by_name('meson.build').list(
                    Project.Cpp.root_directory)
            }

    class Documentation:
        """
        Provides paths within the project's documentation.

        Attributes:
            root_directory: The path to the documentation's root directory
        """

        root_directory = 'doc'

        apidoc_directory = path.join(root_directory, 'developer_guide', 'api')

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
        Provides paths within the project's GitHub-related files.

        Attributes:
            root_directory: The path to the root directory that contains all GitHub-related files
        """

        root_directory = '.github'
