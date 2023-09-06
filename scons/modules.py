"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides access to directories and files belonging to different modules that are part of the project.
"""
from abc import ABC, abstractmethod
from glob import glob
from os import path, walk
from typing import Callable, List


def find_files_recursively(directory: str,
                           directory_filter: Callable[[str], bool] = lambda _: True,
                           file_filter: Callable[[str], bool] = lambda _: True) -> List[str]:
    """
    Finds and returns files in a directory and its subdirectories that match a given filter.

    :param directory:           The directory to be searched
    :param directory_filter:    A function to be used for filtering subdirectories
    :param file_filter:         A function to be used for filtering files
    :return:                    A list that contains the paths of all files that have been found
    """
    result = []

    for root_directory, subdirectories, files in walk(directory, topdown=True):
        subdirectories[:] = [subdirectory for subdirectory in subdirectories if directory_filter(subdirectory)]

        for file in files:
            if file_filter(file):
                result.append(path.join(root_directory, file))

    return result


class Module(ABC):
    """
    An abstract base class for all classes that provide access to directories and files that belong to a module.
    """

    @property
    @abstractmethod
    def root_dir(self) -> str:
        """
        The path to the module's root directory.
        """

    @property
    def build_dir(self) -> str:
        """
        The path to the directory, where build files are stored.
        """
        return path.join(self.root_dir, 'build')

    @property
    def requirements_file(self) -> str:
        """
        The path to the requirements.txt file that specifies dependencies required by a module.
        """
        return path.join(self.root_dir, 'requirements.txt')


class PythonModule(Module):
    """
    Provides access to directories and files that belong to the project's Python code.
    """

    class Subproject:
        """
        Provides access to directories and files that belong to an individual subproject that is part of the project's
        Python code.
        """

        @staticmethod
        def __filter_pycache_directories(directory: str) -> bool:
            return directory != '__pycache__'

        def __init__(self, root_dir: str):
            """
            :param root_dir: The root directory of the subproject
            """
            self.root_dir = root_dir

        @property
        def name(self) -> str:
            """
            The name of the subproject.
            """
            return path.basename(self.root_dir)

        @property
        def source_dir(self) -> str:
            """
            The directory that contains the subproject's source code.
            """
            return path.join(self.root_dir, 'mlrl')

        @property
        def dist_dir(self) -> str:
            """
            The directory that contains all wheel packages that have been built for the subproject.
            """
            return path.join(self.root_dir, 'dist')

        @property
        def build_dirs(self) -> List[str]:
            """
            A list that contains all directories, where the subproject's build files are stored.
            """
            return [self.dist_dir, path.join(self.root_dir, 'build')] + glob(path.join(self.root_dir, '*.egg-info'))

        def find_wheels(self) -> List[str]:
            """
            Finds and returns all wheel packages that have been built for the subproject.

            :return: A list that contains the paths of the wheel packages that have been found
            """
            return glob(path.join(self.dist_dir, '*.whl'))

        def find_source_files(self) -> List[str]:
            """
            Finds and returns all source files that are contained by the subproject.

            :return: A list that contains the paths of the source files that have been found
            """
            return find_files_recursively(self.source_dir, directory_filter=self.__filter_pycache_directories)

        def find_shared_libraries(self) -> List[str]:
            """
            Finds and returns all shared libraries that are contained in the subproject's source tree.

            :return: A list that contains all shared libraries that have been found
            """

            def file_filter(file) -> bool:
                return (file.startswith('lib') and file.find('.so') >= 0) \
                    or file.endswith('.dylib') \
                    or (file.startswith('mlrl') and file.endswith('.lib')) \
                    or file.endswith('.dll')

            return find_files_recursively(self.source_dir,
                                          directory_filter=self.__filter_pycache_directories,
                                          file_filter=file_filter)

        def find_extension_modules(self) -> List[str]:
            """
            Finds and returns all extension modules that are contained in the subproject's source tree.

            :return: A list that contains all extension modules that have been found
            """

            def file_filter(file) -> bool:
                return (not file.startswith('lib') and file.endswith('.so')) \
                    or file.endswith('.pyd') \
                    or (not file.startswith('mlrl') and file.endswith('.lib'))

            return find_files_recursively(self.source_dir,
                                          directory_filter=self.__filter_pycache_directories,
                                          file_filter=file_filter)

    @property
    def root_dir(self) -> str:
        return 'python'

    def find_subprojects(self) -> List[Subproject]:
        """
        Finds and returns all subprojects that are part of the Python code.

        :return: A list that contains all subrojects that have been found
        """
        return [
            PythonModule.Subproject(file) for file in glob(path.join(self.root_dir, 'subprojects', '*'))
            if path.isdir(file)
        ]

    def find_subproject(self, file: str) -> Subproject:
        """
        Finds and returns the subproject to which a given file belongs.

        :param file:    The path of the file
        :return:        The subproject to which the given file belongs
        """
        for subproject in self.find_subprojects():
            if file.startswith(subproject.root_dir):
                return subproject

        raise ValueError('File "' + file + '" does not belong to a Python subproject')


class CppModule(Module):
    """
    Provides access to directories and files that belong to the project's C++ code.
    """

    @property
    def root_dir(self) -> str:
        return 'cpp'


class BuildModule(Module):
    """
    Provides access to directories and files that belong to the build system.
    """

    @property
    def root_dir(self) -> str:
        return 'scons'


BUILD_MODULE = BuildModule()

PYTHON_MODULE = PythonModule()

CPP_MODULE = CppModule()
