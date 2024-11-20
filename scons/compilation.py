"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from os import environ
from typing import List, Optional

from modules import BUILD_MODULE, CPP_MODULE, PYTHON_MODULE
from util.env import get_env
from util.pip import RequirementsFile
from util.run import Program


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

    def as_arguments(self) -> List[str]:
        """
        Returns a list of arguments to be passed to the command "meson configure" for setting the build options.

        :return: A list of arguments
        """
        arguments = []

        for build_option in self.build_options:
            value = build_option.value

            if value:
                arguments.append('-D')
                arguments.append(build_option.key + '=' + value)

        return arguments

    def __bool__(self) -> bool:
        for build_option in self.build_options:
            if build_option.value:
                return True

        return False


CPP_BUILD_OPTIONS = BuildOptions() \
        .add(name='subprojects') \
        .add(name='test_support', subpackage='common') \
        .add(name='multi_threading_support', subpackage='common') \
        .add(name='gpu_support', subpackage='common')


CYTHON_BUILD_OPTIONS = BuildOptions() \
        .add(name='subprojects')


class MesonSetup(Program):
    """
    Allows to run the external program "meson setup".
    """

    def __init__(self, build_directory: str, source_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory:     The path to the build directory
        :param source_directory:    The path to the source directory
        :param build_options:       The build options to be used
        """
        super().__init__(RequirementsFile(BUILD_MODULE.requirements_file), 'meson', 'setup',
                         *build_options.as_arguments(), build_directory, source_directory)
        self.print_arguments(True)


class MesonConfigure(Program):
    """
    Allows to run the external program "meson configure".
    """

    def __init__(self, build_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory: The path to the build directory
        :param build_options:   The build options to be used
        """
        super().__init__(RequirementsFile(BUILD_MODULE.requirements_file), 'meson', 'configure',
                         *build_options.as_arguments(), build_directory)
        self.print_arguments(True)
        self.build_options = build_options

    def run(self):
        if self.build_options:
            print('Configuring build options according to environment variables...')
            super().run()


class MesonCompile(Program):
    """
    Allows to run the external program "meson compile".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__(RequirementsFile(BUILD_MODULE.requirements_file), 'meson', 'compile', '-C', build_directory)
        self.print_arguments(True)


class MesonInstall(Program):
    """
    Allows to run the external program "meson install".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__(RequirementsFile(BUILD_MODULE.requirements_file), 'meson', 'install', '--no-rebuild',
                         '--only-changed', '-C', build_directory)
        self.print_arguments(True)


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    MesonSetup(build_directory=CPP_MODULE.build_dir,
               source_directory=CPP_MODULE.root_dir,
               build_options=CPP_BUILD_OPTIONS) \
        .add_dependencies('ninja') \
        .run()


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    MesonConfigure(CPP_MODULE.build_dir, CPP_BUILD_OPTIONS).run()
    print('Compiling C++ code...')
    MesonCompile(CPP_MODULE.build_dir).run()


def install_cpp(**_):
    """
    Installs shared libraries into the source tree.
    """
    print('Installing shared libraries into source tree...')
    MesonInstall(CPP_MODULE.build_dir).run()


def setup_cython(**_):
    """
    Sets up the build system for compiling the Cython code.
    """
    MesonSetup(build_directory=PYTHON_MODULE.build_dir,
               source_directory=PYTHON_MODULE.root_dir,
               build_options=CYTHON_BUILD_OPTIONS) \
        .add_dependencies('cython') \
        .run()


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    MesonConfigure(PYTHON_MODULE.build_dir, CYTHON_BUILD_OPTIONS)
    print('Compiling Cython code...')
    MesonCompile(PYTHON_MODULE.build_dir).run()


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    MesonInstall(PYTHON_MODULE.build_dir).run()
