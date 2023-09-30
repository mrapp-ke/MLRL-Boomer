"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from environment import get_string_env
from modules import CPP_MODULE, PYTHON_MODULE
from run import run_venv_program


class BuildOptions:
    """
    Allows to obtain build options from environment variables.
    """

    class BuildOption(ABC):
        """
        A single build option.
        """

        def __init__(self, subpackage: str, name: str):
            """
            :param subpackage:  The subpackage, the build option corresponds to
            :param name:        The name of the build option
            """
            self.subpackage = subpackage
            self.name = name

        @abstractmethod
        def get_value(self) -> Optional[str]:
            """
            Returns the value to be set for the build option.

            :return: The value to be set or None, if no value should be set
            """

    class FeatureBuildOption(BuildOption):
        """
        A single build option for enabling or disabling a feature at compile-time.
        """

        def get_value(self) -> Optional[str]:
            return get_string_env(self.name.upper(), accepted_values={'enabled', 'disabled'})

    def __init__(self):
        self.build_options = []

    def add_feature(self, subpackage: str, name: str) -> 'BuildOptions':
        """
        Adds a build option for enabling or disabling a feature at compile-time.

        :param subpackage:  The subpackage, the build option corresponds to
        :param name:        The name of the build option
        :return:            The `BuildOptions` itself
        """
        self.build_options.append(BuildOptions.FeatureBuildOption(subpackage=subpackage, name=name))
        return self

    def to_args(self) -> List[str]:
        """
        Returns a list of arguments to be passed to the command "meson configure" for setting the build options.

        :return: A list of arguments
        """
        args = []

        for build_option in self.build_options:
            value = build_option.get_value()

            if value:
                args.append('-D')
                args.append(build_option.subpackage + ':' + build_option.name + '=' + value)

        return args


def __meson_setup(root_dir: str, build_dir: str, dependencies: Optional[List[str]] = None):
    print('Setting up build directory "' + build_dir + '"...')
    run_venv_program('meson', 'setup', build_dir, root_dir, print_args=True, additional_dependencies=dependencies)


def __meson_configure(build_dir: str, build_options: BuildOptions):
    args = build_options.to_args()

    if args:
        print('Configuring build according to environment variables...')
        run_venv_program('meson', 'configure', *args, build_dir, print_args=True)


def __meson_compile(build_dir: str):
    run_venv_program('meson', 'compile', '-C', build_dir, print_args=True)


def __meson_install(build_dir: str):
    run_venv_program('meson', 'install', '--no-rebuild', '--only-changed', '-C', build_dir, print_args=True)


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    __meson_setup(CPP_MODULE.root_dir, CPP_MODULE.build_dir, dependencies=['ninja'])


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    build_options = BuildOptions() \
        .add_feature(subpackage='common', name='test_support') \
        .add_feature(subpackage='common', name='multi_threading_support') \
        .add_feature(subpackage='common', name='gpu_support')
    __meson_configure(CPP_MODULE.build_dir, build_options)
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
    __meson_setup(PYTHON_MODULE.root_dir, PYTHON_MODULE.build_dir, dependencies=['cython'])


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    print('Compiling Cython code...')
    __meson_compile(PYTHON_MODULE.build_dir)


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    __meson_install(PYTHON_MODULE.build_dir)
