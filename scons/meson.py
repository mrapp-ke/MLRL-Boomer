"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "meson".
"""
from abc import ABC
from typing import List

from compilation.build_options import BuildOptions
from util.run import Program


def build_options_as_meson_arguments(build_options: BuildOptions) -> List[str]:
    """
    Returns a list of arguments that can be passed to meson for setting build options.

    :param build_options:   The build options
    :return:                A list of arguments
    """
    arguments = []

    for build_option in build_options:
        if build_option:
            arguments.append('-D')
            arguments.append(build_option.key + '=' + build_option.value)

    return arguments


class Meson(Program, ABC):
    """
    An abstract base class for all classes that allow to run the external program "meson".
    """

    def __init__(self, meson_command: str, *arguments: str):
        """
        :param program:     The meson command to be run
        :param arguments:   Optional arguments to be passed to meson
        """
        super().__init__('meson', meson_command, *arguments)
        self.print_arguments(True)


class MesonSetup(Meson):
    """
    Allows to run the external program "meson setup".
    """

    def __init__(self, build_directory: str, source_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory:     The path to the build directory
        :param source_directory:    The path to the source directory
        :param build_options:       The build options to be used
        """
        super().__init__('setup', *build_options_as_meson_arguments(build_options), build_directory, source_directory)


class MesonConfigure(Meson):
    """
    Allows to run the external program "meson configure".
    """

    def __init__(self, build_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory: The path to the build directory
        :param build_options:   The build options to be used
        """
        super().__init__('configure', *build_options_as_meson_arguments(build_options), build_directory)
        self.build_options = build_options

    def _should_be_skipped(self) -> bool:
        return not self.build_options

    def _before(self):
        print('Configuring build options according to environment variables...')


class MesonCompile(Meson):
    """
    Allows to run the external program "meson compile".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__('compile', '-C', build_directory)


class MesonInstall(Meson):
    """
    Allows to run the external program "meson install".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__('install', '--no-rebuild', '--only-changed', '-C', build_directory)
