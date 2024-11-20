"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running external programs during the build process.
"""
from subprocess import CompletedProcess

from util.cmd import Command
from util.pip import Pip


class Program(Command):
    """
    Allows to run an external program.
    """

    class RunOptions(Command.RunOptions):
        """
        Allows to customize options for running an external program.
        """

        def __init__(self, requirements_file: RequirementsFile):
            """
            :param requirements_file: The requirements file that should be used for looking up dependency versions
            """
            super().__init__()
            self.requirements_file = requirements_file
            self.install_program = True
            self.dependencies = set()

        def run(self, command: Command, capture_output: bool) -> CompletedProcess:
            dependencies = []

            if self.install_program:
                dependencies.append(command.command)

            dependencies.extend(self.dependencies)
            Pip(self.requirements_file).install_packages(*dependencies)
            return super().run(command, capture_output)

    def __init__(self, requirements_file: RequirementsFile, program: str, *arguments: str):
        """
        :param requirements_file:   The requirements file that should be used for looking up dependency versions
        :param program:             The name of the program to be run
        :param arguments:           Optional arguments to be passed to the program
        """
        super().__init__(program, *arguments, run_options=Program.RunOptions(requirements_file))

    def install_program(self, install_program: bool) -> 'Program':
        """
        Sets whether the program should be installed via pip before being run or not.

        :param install_program: True, if the program should be installed before being run, False otherwise
        :return:                The `Program` itself
        """
        self.run_options.install_program = install_program
        return self

    def add_dependencies(self, *dependencies: str) -> 'Program':
        """
        Adds one or several Python packages that should be installed before running the program.

        :param dependencies:    The names of the Python packages to be added
        :return:                The `Program` itself
        """
        self.run_options.dependencies.update(dependencies)
        return self


class PythonModule(Program):
    """
    Allows to run a Python module.
    """

    def __init__(self, requirements_file: RequirementsFile, module: str, *arguments: str):
        """
        :param requirements_file:   The requirements file that should be used for looking up dependency versions
        :param module:              The name of the module to be run
        :param arguments:           Optional arguments to be passed to the module
        """
        super().__init__(requirements_file, 'python', '-m', module, *arguments)
        self.module = module
        self.install_program(True)

    def install_program(self, install_program: bool) -> Program:
        super().install_program(False)

        if install_program:
            super().add_dependencies(self.module)
        else:
            self.run_options.dependencies.remove(self.module)

        return self
