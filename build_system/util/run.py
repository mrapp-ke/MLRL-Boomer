"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running external programs during the build process.
"""
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Set, override

from core.build_unit import BuildUnit
from util.cmd import Command
from util.package_manager import PackageManager
from util.requirements import RequirementsFiles


class Program(Command):
    """
    Allows to run an external program.
    """

    class RunOptions(Command.RunOptions):
        """
        Allows to customize options for running an external program.
        """

        def __init__(self, build_unit: BuildUnit = BuildUnit.for_file(Path(__file__))):
            """
            :param build_unit: The build unit from which the program should be run
            """
            super().__init__()
            self.build_unit = build_unit
            self.install_program = True
            self.install_silent = False
            self.dependencies: Set[str] = set()

        @override
        def run(self, command: Command, capture_output: bool) -> CompletedProcess[Any]:
            dependencies = []

            if self.install_program:
                dependencies.append(command.command)

            dependencies.extend(self.dependencies)
            PackageManager.install_packages(RequirementsFiles.for_build_unit(self.build_unit),
                                            *dependencies,
                                            silent=self.install_silent)
            return super().run(command, capture_output)

    def __init__(self, program: str, *arguments: str):
        """
        :param program:     The name of the program to be run
        :param arguments:   Optional arguments to be passed to the program
        """
        super().__init__(program, *arguments)
        self.program_run_options = Program.RunOptions()
        self.run_options = self.program_run_options

    def set_build_unit(self, build_unit: BuildUnit) -> 'Program':
        """
        Sets the build unit from which the program should be run.

        :param build_unit:  The build unit to be set
        :return:            The `Program` itself
        """
        self.program_run_options.build_unit = build_unit
        return self

    def install_program(self, install_program: bool, silent: bool = False) -> 'Program':
        """
        Sets whether the program should be installed via the package manager before being run or not.

        :param install_program: True, if the program should be installed before being run, False otherwise
        :param silent:          True, if any log output should be suppressed, False otherwise
        :return:                The `Program` itself
        """
        self.program_run_options.install_program = install_program
        self.program_run_options.install_silent = silent
        return self

    def add_dependencies(self, *dependencies: str) -> 'Program':
        """
        Adds one or several Python packages that should be installed before running the program.

        :param dependencies:    The names of the Python packages to be added
        :return:                The `Program` itself
        """
        self.program_run_options.dependencies.update(dependencies)
        return self

    def set_accepted_exit_codes(self, *accepted_exit_codes: int) -> 'Program':
        """
        Sets one or several exit codes that should not be considered as an error.

        :param accepted_exit_codes: The exit codes to be set
        :return:                    The `Program` itself
        """
        self.run_options.accepted_exit_codes = set(accepted_exit_codes)
        return self


class PythonModule(Program):
    """
    Allows to run a Python module.
    """

    def __init__(self, module: str, *arguments: str):
        """
        :param module:      The name of the module to be run
        :param arguments:   Optional arguments to be passed to the module
        """
        super().__init__('python', '-m', module, *arguments)
        self.module = module
        self.install_program(True)

    @override
    def install_program(self, install_program: bool, silent: bool = False) -> Program:
        super().install_program(False, silent=silent)

        if install_program:
            super().add_dependencies(self.module)
        else:
            self.program_run_options.dependencies.remove(self.module)

        return self
