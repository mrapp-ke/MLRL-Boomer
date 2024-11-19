"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from os import environ
from typing import List, Optional

from modules import CPP_MODULE, PYTHON_MODULE
from run import run_program
from util.env import get_env


class BuildOptions:
    """
    Allows to obtain build options from environment variables.
    """

    class BuildOption:
        """
        A single build option.
        """

        def __init__(self, name: str, subpackage: Optional[str] = None):
            """
            :param name:        The name of the build option
            :param subpackage:  The subpackage, the build option corresponds to, or None, if it is a global option
            """
            self.name = name
            self.subpackage = subpackage

        @property
        def key(self) -> str:
            """
            The key to be used for setting the build option.
            """
            return (self.subpackage + ':' if self.subpackage else '') + self.name

        @property
        def value(self) -> Optional[str]:
            """
            Returns the value to be set for the build option.

            :return: The value to be set or None, if no value should be set
            """
            value = get_env(environ, self.name.upper(), None)

            if value:
                value = value.strip()

            return value

    def __init__(self):
        self.build_options = []

    def add(self, name: str, subpackage: Optional[str] = None) -> 'BuildOptions':
        """
        Adds a build option.

        :param name:        The name of the build option
        :param subpackage:  The subpackage, the build option corresponds to, or None, if it is a global option
        :return:            The `BuildOptions` itself
        """
        self.build_options.append(BuildOptions.BuildOption(name=name, subpackage=subpackage))
        return self

    def to_args(self) -> List[str]:
        """
        Returns a list of arguments to be passed to the command "meson configure" for setting the build options.

        :return: A list of arguments
        """
        args = []

        for build_option in self.build_options:
            value = build_option.value

            if value:
                args.append('-D')
                args.append(build_option.key + '=' + value)

        return args


CPP_BUILD_OPTIONS = BuildOptions() \
        .add(name='subprojects') \
        .add(name='test_support', subpackage='common') \
        .add(name='multi_threading_support', subpackage='common') \
        .add(name='gpu_support', subpackage='common')


CYTHON_BUILD_OPTIONS = BuildOptions() \
        .add(name='subprojects')


def __meson_setup(root_dir: str,
                  build_dir: str,
                  build_options: BuildOptions = BuildOptions(),
                  dependencies: Optional[List[str]] = None):
    print('Setting up build directory "' + build_dir + '"...')
    args = build_options.to_args()
    run_program('meson', 'setup', *args, build_dir, root_dir, print_args=True, additional_dependencies=dependencies)


def __meson_configure(build_dir: str, build_options: BuildOptions):
    args = build_options.to_args()

    if args:
        print('Configuring build options according to environment variables...')
        run_program('meson', 'configure', *args, build_dir, print_args=True)


def __meson_compile(build_dir: str):
    run_program('meson', 'compile', '-C', build_dir, print_args=True)


def __meson_install(build_dir: str):
    run_program('meson', 'install', '--no-rebuild', '--only-changed', '-C', build_dir, print_args=True)


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    __meson_setup(CPP_MODULE.root_dir, CPP_MODULE.build_dir, CPP_BUILD_OPTIONS, dependencies=['ninja'])


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    __meson_configure(CPP_MODULE.build_dir, CPP_BUILD_OPTIONS)
    print('Compiling C++ code...')
    __meson_compile(CPP_MODULE.build_dir)


def install_cpp(**_):
    """
    Installs shared libraries into the source tree.
    """
    print('Installing shared libraries into source tree...')
    __meson_install(CPP_MODULE.build_dir)


def setup_cython(**_):
    """
    Sets up the build system for compiling the Cython code.
    """
    __meson_setup(PYTHON_MODULE.root_dir, PYTHON_MODULE.build_dir, CYTHON_BUILD_OPTIONS, dependencies=['cython'])


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    __meson_configure(PYTHON_MODULE.build_dir, CYTHON_BUILD_OPTIONS)
    print('Compiling Cython code...')
    __meson_compile(PYTHON_MODULE.build_dir)


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    __meson_install(PYTHON_MODULE.build_dir)
