"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines modules to be dealt with by the build system.
"""
from code_style.modules import CodeModule
from dependencies.python.modules import DependencyType, PythonDependencyModule
from util.files import FileSearch
from util.languages import Language

MODULES = [
    CodeModule(Language.YAML, '.',
               FileSearch().set_recursive(False).set_hidden(True)),
    CodeModule(Language.YAML, '.github'),
    CodeModule(Language.MARKDOWN, '.',
               FileSearch().set_recursive(False)),
    CodeModule(Language.MARKDOWN, 'doc'),
    CodeModule(Language.MARKDOWN, 'python'),
    CodeModule(Language.PYTHON, 'scons'),
    CodeModule(Language.PYTHON, 'doc'),
    CodeModule(Language.PYTHON, 'python',
               FileSearch().set_recursive(True).exclude_subdirectories_by_name('build')),
    CodeModule(Language.CYTHON, 'python',
               FileSearch().set_recursive(True).exclude_subdirectories_by_name('build')),
    CodeModule(Language.CPP, 'cpp',
               FileSearch().set_recursive(True).exclude_subdirectories_by_name('build')),
    PythonDependencyModule(DependencyType.BUILD_TIME, 'scons',
                           FileSearch().set_recursive(True)),
    PythonDependencyModule(DependencyType.BUILD_TIME, 'doc',
                           FileSearch().set_recursive(True)),
    PythonDependencyModule(DependencyType.RUNTIME, 'python')
]
