"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides paths within the project that are important for the build system.
"""
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

        root_directory = 'scons'

        build_directory_name = 'build'

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
            root_directory:         The path to the Python code's root directory
            build_directory_name:   The name of the Python code's build directory
            test_directory_name:                The name fo the directory that contains tests
        """

        root_directory = 'python'

        build_directory_name = 'build'

        test_directory_name = 'tests'

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searching for files within the Python code.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name(Project.Python.build_directory_name, 'dist', '__pycache__') \
                .exclude_subdirectories_by_substrings(ends_with='.egg.info')

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

    class Documentation:
        """
        Provides paths within the project's documentation.

        Attributes:
            root_directory: The path to the documentation's root directory
        """

        root_directory = 'doc'

        @staticmethod
        def file_search() -> FileSearch:
            """
            Creates and returns a `FileSearch` that allows searching for files within the documentation.

            :return: The `FileSearch` that has been created
            """
            return FileSearch() \
                .set_recursive(True) \
                .exclude_subdirectories_by_name('_build')

    class Github:
        """
        Provides paths within the project's GitHub-related files.

        Attributes:
            root_directory: The path to the root directory that contains all GitHub-related files
        """

        root_directory = '.github'
