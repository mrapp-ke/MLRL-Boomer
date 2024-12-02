"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "meson".
"""
from abc import ABC
from typing import List

from compilation.build_options import BuildOptions
from compilation.modules import CompilationModule
from util.run import Program
from util.units import BuildUnit


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

    def __init__(self, build_unit: BuildUnit, meson_command: str, *arguments: str):
        """
        :param build_unit:  The build unit from which the program should be run
        :param program:     The meson command to be run
        :param arguments:   Optional arguments to be passed to meson
        """
        super().__init__('meson', meson_command, *arguments)
        self.print_arguments(True)
        self.set_build_unit(build_unit)


class MesonSetup(Meson):
    """
    Allows to run the external program "meson setup".
    """

    def __init__(self, build_unit: BuildUnit, module: CompilationModule, build_options: BuildOptions = BuildOptions()):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param build_options:   The build options to be used
        """
        super().__init__(build_unit, 'setup', *build_options_as_meson_arguments(build_options), module.build_directory,
                         module.root_directory)
        self.add_dependencies('ninja')


class MesonConfigure(Meson):
    """
    Allows to run the external program "meson configure".
    """

    def __init__(self, build_unit: BuildUnit, module: CompilationModule, build_options: BuildOptions = BuildOptions()):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param build_options:   The build options to be used
        """
        super().__init__(build_unit, 'configure', *build_options_as_meson_arguments(build_options),
                         module.build_directory)
        self.build_options = build_options

    def _should_be_skipped(self) -> bool:
        return not self.build_options

    def _before(self):
        print('Configuring build options according to environment variables...')


class MesonCompile(Meson):
    """
    Allows to run the external program "meson compile".
    """

    def __init__(self, build_unit: BuildUnit, module: CompilationModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit, 'compile', '-C', module.build_directory)


class MesonInstall(Meson):
    """
    Allows to run the external program "meson install".
    """

    def __init__(self, build_unit: BuildUnit, module: CompilationModule):
        """
        :param build_unit:  The build unit from which the program should be run
        :param module:      The module, the program should be applied to
        """
        super().__init__(build_unit, 'install', '--no-rebuild', '--only-changed', '-C', module.build_directory)
        self.install_program(False)
