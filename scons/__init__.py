"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines modules to be dealt with by the build system.
"""
from code_style.modules import CodeModule
from compilation.modules import CompilationModule
from dependencies.python.modules import DependencyType, PythonDependencyModule
from testing.cpp.modules import CppTestModule
from testing.python.modules import PythonTestModule
from util.files import FileSearch
from util.languages import Language

MODULES = [
    CodeModule(
        language=Language.YAML,
        root_directory='.',
        file_search=FileSearch().set_recursive(False).set_hidden(True),
    ),
    CodeModule(
        language=Language.YAML,
        root_directory='.github',
    ),
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='.',
        file_search=FileSearch().set_recursive(False),
    ),
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='doc',
    ),
    CodeModule(
        language=Language.MARKDOWN,
        root_directory='python',
    ),
    CodeModule(
        language=Language.PYTHON,
        root_directory='scons',
    ),
    CodeModule(
        language=Language.PYTHON,
        root_directory='doc',
    ),
    CodeModule(
        language=Language.PYTHON,
        root_directory='python',
        file_search=FileSearch() \
            .set_recursive(True) \
            .exclude_subdirectories_by_name('build', 'dist', '__pycache__') \
            .exclude_subdirectories_by_substrings(ends_with='.egg.info'),
    ),
    CodeModule(
        language=Language.CYTHON,
        root_directory='python',
        file_search=FileSearch() \
            .set_recursive(True) \
            .exclude_subdirectories_by_name('build', 'dist', '__pycache__') \
            .exclude_subdirectories_by_substrings(ends_with='.egg-info'),
    ),
    CodeModule(
        language=Language.CPP,
        root_directory='cpp',
        file_search=FileSearch().set_recursive(True).exclude_subdirectories_by_name('build'),
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory='scons',
        file_search=FileSearch().set_recursive(True),
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.BUILD_TIME,
        root_directory='doc',
        file_search=FileSearch().set_recursive(True),
    ),
    PythonDependencyModule(
        dependency_type=DependencyType.RUNTIME,
        root_directory='python',
    ),
    CompilationModule(
        language=Language.CPP,
        root_directory='cpp',
        install_directory='python',
        file_search=FileSearch() \
            .filter_by_substrings(starts_with='lib', contains='.so') \
            .filter_by_substrings(ends_with='.dylib') \
            .filter_by_substrings(starts_with='mlrl', ends_with='.lib') \
            .filter_by_substrings(ends_with='.dll'),
    ),
    CompilationModule(
        language=Language.CYTHON,
        root_directory='python',
        file_search=FileSearch() \
            .filter_by_substrings(not_starts_with='lib', ends_with='.so') \
            .filter_by_substrings(ends_with='.pyd') \
            .filter_by_substrings(not_starts_with='mlrl', ends_with='.lib'),
    ),
    CppTestModule(
        root_directory='cpp',
    ),
    PythonTestModule(
        root_directory='python',
    )
]
