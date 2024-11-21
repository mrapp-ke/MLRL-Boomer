"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides access to directories and files belonging to different modules that are part of the project.
"""
from abc import ABC, abstractmethod
from glob import glob
from os import environ, path, walk
from typing import Callable, List, Optional

from util.env import get_env_array
from util.units import BuildUnit


def find_files_recursively(directory: str,
                           directory_filter: Callable[[str, str], bool] = lambda *_: True,
                           file_filter: Callable[[str, str], bool] = lambda *_: True) -> List[str]:
    """
    Finds and returns files in a directory and its subdirectories that match a given filter.

    :param directory:           The directory to be searched
    :param directory_filter:    A function to be used for filtering subdirectories
    :param file_filter:         A function to be used for filtering files
    :return:                    A list that contains the paths of all files that have been found
    """
    result = []

    for parent_directory, subdirectories, files in walk(directory, topdown=True):
        subdirectories[:] = [
            subdirectory for subdirectory in subdirectories if directory_filter(parent_directory, subdirectory)
        ]

        for file in files:
            if file_filter(parent_directory, file):
                result.append(path.join(parent_directory, file))

    return result


class Module(BuildUnit, ABC):
    """
    An abstract base class for all classes that provide access to directories and files that belong to a module.
    """

    def __init__(self):
        super().__init__(path.join(self.root_dir, 'requirements.txt'))

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


class SourceModule(Module, ABC):
    """
    An abstract base class for all classes that provide access to directories and files that belong to a module, which
    contains source code.
    """

    class Subproject(ABC):
        """
        An abstract base class for all classes that provide access to directories and files that belong to an individual
        subproject that is part of a module, which contains source files.
        """

        def __init__(self, parent_module: 'SourceModule', root_dir: str):
            """
            :param parent_module:   The `SourceModule`, the subproject belongs to
            :param root_dir:        The root directory of the suproject
            """
            self.parent_module = parent_module
            self.root_dir = root_dir

        @property
        def name(self) -> str:
            """
            The name of the subproject.
            """
            return path.basename(self.root_dir)

        def is_enabled(self) -> bool:
            """
            Returns whether the subproject is enabled or not.

            :return: True, if the subproject is enabled, False otherwise
            """
            enabled_subprojects = get_env_array(environ, 'SUBPROJECTS')
            return not enabled_subprojects or self.name in enabled_subprojects


class PythonModule(SourceModule):
    """
    Provides access to directories and files that belong to the project's Python code.
    """

    class Subproject(SourceModule.Subproject):
        """
        Provides access to directories and files that belong to an individual subproject that is part of the project's
        Python code.
        """

        @staticmethod
        def __filter_pycache_directories(_: str, directory: str) -> bool:
            return directory != '__pycache__'

        @property
        def source_dir(self) -> str:
            """
            The directory that contains the subproject's source code.
            """
            return path.join(self.root_dir, 'mlrl')

        @property
        def test_dir(self) -> str:
            """
            The directory that contains the subproject's automated tests.
            """
            return path.join(self.root_dir, 'tests')

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

            def file_filter(_: str, file: str) -> bool:
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

            def file_filter(_: str, file: str) -> bool:
                return (not file.startswith('lib') and file.endswith('.so')) \
                    or file.endswith('.pyd') \
                    or (not file.startswith('mlrl') and file.endswith('.lib'))

            return find_files_recursively(self.source_dir,
                                          directory_filter=self.__filter_pycache_directories,
                                          file_filter=file_filter)

    @property
    def root_dir(self) -> str:
        return 'python'

    def find_subprojects(self, return_all: bool = False) -> List[Subproject]:
        """
        Finds and returns all subprojects that are part of the Python code.

        :param return_all:  True, if all subprojects should be returned, even if they are disabled, False otherwise
        :return:            A list that contains all subrojects that have been found
        """
        subprojects = [
            PythonModule.Subproject(self, file) for file in glob(path.join(self.root_dir, 'subprojects', '*'))
            if path.isdir(file)
        ]
        return subprojects if return_all else [subproject for subproject in subprojects if subproject.is_enabled()]

    def find_subproject(self, file: str) -> Optional[Subproject]:
        """
        Finds and returns the subproject to which a given file belongs.

        :param file:    The path of the file
        :return:        The subproject to which the given file belongs or None, if no such subproject is available
        """
        for subproject in self.find_subprojects():
            if file.startswith(subproject.root_dir):
                return subproject

        return None


class CppModule(SourceModule):
    """
    Provides access to directories and files that belong to the project's C++ code.
    """

    class Subproject(SourceModule.Subproject):
        """
        Provides access to directories and files that belong to an individual subproject that is part of the project's
        C++ code.
        """

        @property
        def include_dir(self) -> str:
            """
            The directory that contains the header files.
            """
            return path.join(self.root_dir, 'include')

        @property
        def src_dir(self) -> str:
            """
            The directory that contains the source files.
            """
            return path.join(self.root_dir, 'src')

        @property
        def test_dir(self) -> str:
            """
            The directory that contains the source code for automated tests.
            """
            return path.join(self.root_dir, 'test')

        def find_source_files(self) -> List[str]:
            """
            Finds and returns all source files that are contained by the subproject.

            :return: A list that contains the paths of the source files that have been found
            """

            def file_filter(_: str, file: str) -> bool:
                return file.endswith('.hpp') or file.endswith('.cpp')

            return find_files_recursively(self.root_dir, file_filter=file_filter)

    @property
    def root_dir(self) -> str:
        return 'cpp'

    def find_subprojects(self, return_all: bool = False) -> List[Subproject]:
        """
        Finds and returns all subprojects that are part of the C++ code.

        
        :param return_all:  True, if all subprojects should be returned, even if they are disabled, False otherwise
        :return:            A list that contains all subprojects that have been found
        """
        subprojects = [
            CppModule.Subproject(self, file) for file in glob(path.join(self.root_dir, 'subprojects', '*'))
            if path.isdir(file)
        ]
        return subprojects if return_all else [subproject for subproject in subprojects if subproject.is_enabled()]


class BuildModule(Module):
    """
    Provides access to directories and files that belong to the build system.
    """

    @property
    def root_dir(self) -> str:
        return 'scons'


class DocumentationModule(Module):
    """
    Provides access to directories and files that belong to the project's documentation.
    """

    class ApidocSubproject(ABC):
        """
        An abstract base class for all classes that provide access to directories and files that are needed for building
        the API documentation of a certain C++ or Python subproject.
        """

        def __init__(self, parent_module: 'DocumentationModule', source_subproject: SourceModule.Subproject):
            """
            :param parent_module:       The `DocumentationModule` this subproject belongs to
            :param source_subproject:   The subproject of which the API documentation should be built
            """
            self.parent_module = parent_module
            self.source_subproject = source_subproject

        @property
        def name(self) -> str:
            """
            The name of the subproject of which the API documentation should be built.
            """
            return self.source_subproject.name

        @property
        @abstractmethod
        def build_dir(self) -> str:
            """
            The directory, where build files should be stored.
            """

        @property
        @abstractmethod
        def root_file(self) -> str:
            """
            The path of the root file of the API documentation.
            """

        def find_build_files(self) -> List[str]:
            """
            Finds and returns all build files that have been created when building the API documentation.

            :return: A list that contains the paths of all build files that have been found
            """
            return find_files_recursively(self.build_dir)

    class CppApidocSubproject(ApidocSubproject):
        """
        Provides access to the directories and files that are necessary for building the API documentation of a certain
        C++ subproject.
        """

        @property
        def build_dir(self) -> str:
            return path.join(self.parent_module.apidoc_dir_cpp, self.name)

        @property
        def root_file(self) -> str:
            return path.join(self.build_dir, 'filelist.rst')

    class PythonApidocSubproject(ApidocSubproject):
        """
        Provides access to the directories and files that are necessary for building the API documentation of a certain
        Python subproject.
        """

        @property
        def build_dir(self) -> str:
            return path.join(self.parent_module.apidoc_dir_python, self.name)

        @property
        def root_file(self) -> str:
            return path.join(self.build_dir, 'mlrl.' + self.name + '.rst')

    @property
    def root_dir(self) -> str:
        return 'doc'

    @property
    def doxygen_config_file(self) -> str:
        """
        The Doxygen config file.
        """
        return path.join(self.root_dir, 'Doxyfile')

    @property
    def config_file(self) -> str:
        """
        The config file that should be used for building the documentation.
        """
        return path.join(self.root_dir, 'conf.py')

    @property
    def apidoc_dir(self) -> str:
        """
        The directory, where API documentations should be stored.
        """
        return path.join(self.root_dir, 'developer_guide', 'api')

    @property
    def apidoc_dir_python(self) -> str:
        """
        The directory, where Python API documentations should be stored.
        """
        return path.join(self.apidoc_dir, 'python')

    @property
    def apidoc_tocfile_python(self) -> str:
        """
        The tocfile referencing all Python API documentations.
        """
        return path.join(self.apidoc_dir_python, 'index.md')

    @property
    def apidoc_dir_cpp(self) -> str:
        """
        The directory, where C++ API documentations should be stored.
        """
        return path.join(self.apidoc_dir, 'cpp')

    @property
    def apidoc_tocfile_cpp(self) -> str:
        """
        The tocfile referencing all C++ API documentations.
        """
        return path.join(self.apidoc_dir_cpp, 'index.md')

    @property
    def build_dir(self) -> str:
        """
        The directory, where the documentation should be stored.
        """
        return path.join(self.root_dir, '_build', 'html')

    def find_build_files(self) -> List[str]:
        """
        Finds and returns all files that belong to the documentation that has been built.

        :return: A list that contains the paths of the build files that have been found
        """
        return find_files_recursively(self.build_dir)

    def find_source_files(self) -> List[str]:
        """
        Finds and returns all source files from which the documentation is built.

        :return: A list that contains the paths of the source files that have been found
        """

        def directory_filter(parent_directory: str, directory: str) -> bool:
            return path.join(parent_directory, directory) != self.build_dir

        def file_filter(_: str, file: str) -> bool:
            return file == 'conf.py' or file.endswith('.rst') or file.endswith('.svg') or file.endswith('.md')

        return find_files_recursively(self.root_dir, directory_filter=directory_filter, file_filter=file_filter)

    def get_cpp_apidoc_subproject(self, cpp_subproject: CppModule.Subproject) -> CppApidocSubproject:
        """
        Returns a `CppApidocSubproject` for building the API documentation of a given C++ subproject.

        :param cpp_subproject:  The C++ subproject of which the API documentation should be built
        :return:                A `CppApidocSubproject`
        """
        return DocumentationModule.CppApidocSubproject(self, cpp_subproject)

    def get_python_apidoc_subproject(self, python_subproject: PythonModule.Subproject) -> PythonApidocSubproject:
        """
        Returns a `PythonApidocSubproject` for building the API documentation of a given Python subproject.

        :param python_subproject:   The Python subproject of which the API documentation should be built
        :return:                    A `PythonApidocSubproject`
        """
        return DocumentationModule.PythonApidocSubproject(self, python_subproject)

    def find_cpp_apidoc_subproject(self, file: str) -> Optional[CppApidocSubproject]:
        """
        Finds and returns the `CppApidocSubproject` to which a given file belongs.

        :param file:    The path of the file
        :return:        The `CppApiSubproject` to which the given file belongs or None, if no such subproject is
                        available
        """
        for subproject in CPP_MODULE.find_subprojects():
            apidoc_subproject = self.get_cpp_apidoc_subproject(subproject)

            if file.startswith(apidoc_subproject.build_dir):
                return apidoc_subproject

        return None

    def find_python_apidoc_subproject(self, file: str) -> Optional[PythonApidocSubproject]:
        """
        Finds and returns the `PythonApidocSubproject` to which a given file belongs.

        :param file:    The path of the file
        :return:        The `PythonApidocSubproject` to which the given file belongs or None, if no such subproject is
                        available
        """
        for subproject in PYTHON_MODULE.find_subprojects():
            apidoc_subproject = self.get_python_apidoc_subproject(subproject)

            if file.startswith(apidoc_subproject.build_dir):
                return apidoc_subproject

        return None


BUILD_MODULE = BuildModule()

PYTHON_MODULE = PythonModule()

CPP_MODULE = CppModule()

DOC_MODULE = DocumentationModule()

ALL_MODULES = [BUILD_MODULE, PYTHON_MODULE, CPP_MODULE, DOC_MODULE]
