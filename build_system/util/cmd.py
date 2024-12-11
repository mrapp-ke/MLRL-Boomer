"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for running command line programs during the build process.
"""
import subprocess
import sys

from os import path
from subprocess import CompletedProcess

from util.format import format_iterable
from util.log import Log


class Command:
    """
    Allows to run command line programs.
    """

    class PrintOptions:
        """
        Allows to customize how command line programs are presented in log statements.
        """

        def __init__(self):
            self.print_arguments = False

        def format(self, command: 'Command') -> str:
            """
            Creates and returns a textual representation of a given command line program.

            :param command: The command line program
            :return:        The textual representation that has been created
            """
            result = command.command

            if self.print_arguments:
                result += ' ' + format_iterable(command.arguments, separator=' ')

            return result

    class RunOptions:
        """
        Allows to customize options for running command line programs.
        """

        def __init__(self):
            self.print_command = True
            self.exit_on_error = True
            self.environment = None

        def run(self, command: 'Command', capture_output: bool) -> CompletedProcess:
            """
            Runs a given command line program.

            :param command:         The command line program to be run
            :param capture_output:  True, if the output of the program should be captured, False otherwise
            :return:                The output of the program
            """
            if self.print_command:
                Log.info('Running external command "%s"...', command.print_options.format(command))

            output = subprocess.run([command.command] + command.arguments,
                                    check=False,
                                    text=capture_output,
                                    capture_output=capture_output,
                                    env=self.environment)
            exit_code = output.returncode

            if exit_code != 0:
                message = ('External command "' + str(command) + '" terminated with non-zero exit code '
                           + str(exit_code))

                if self.exit_on_error:
                    if capture_output:
                        Log.info('%s', str(output.stderr).strip())

                    Log.error(message, exit_code=exit_code)
                else:
                    raise RuntimeError(message)

            return output

    @staticmethod
    def __in_virtual_environment() -> bool:
        return sys.prefix != sys.base_prefix

    def __init__(self,
                 command: str,
                 *arguments: str,
                 print_options: PrintOptions = PrintOptions(),
                 run_options: RunOptions = RunOptions()):
        """
        :param command:         The name of the command line program
        :param arguments:       Optional arguments to be passed to the command line program
        :param run_options:     The options that should eb used for running the command line program
        :param print_options:   The options that should be used for creating textual representations of the command line
                                program
        """
        self.command = command

        if self.__in_virtual_environment():
            # On Windows, we use the relative path to the command's executable within the virtual environment, if such
            # an executable exists. This circumvents situations where the PATH environment variable has not been updated
            # after activating the virtual environment. This can prevent the executables from being found or can lead to
            # the wrong executable, from outside the virtual environment, being executed.
            executable = path.join(sys.prefix, 'Scripts', command + '.exe')

            if path.isfile(executable):
                self.command = executable

        self.arguments = list(arguments)
        self.print_options = print_options
        self.run_options = run_options

    def add_arguments(self, *arguments: str) -> 'Command':
        """
        Adds one or several arguments to be passed to the command line program.

        :param arguments:   The arguments to be added
        :return:            The `Command` itself
        """
        self.arguments.extend(arguments)
        return self

    def add_conditional_arguments(self, condition: bool, *arguments: str) -> 'Command':
        """
        Adds one or several arguments to be passed to the command line program, if a certain condition is True.

        :param condition:   The condition
        :param arguments:   The arguments to be added
        :return:            The `Command` itself
        """
        if condition:
            self.arguments.extend(arguments)
        return self

    def print_arguments(self, print_arguments: bool) -> 'Command':
        """
        Sets whether the arguments of the command line program should be included in log statements or not.

        :param print_arguments: True, if the arguments should be included, False otherwise
        :return:                The `Command` itself
        """
        self.print_options.print_arguments = print_arguments
        return self

    def print_command(self, print_command: bool) -> 'Command':
        """
        Sets whether the command line program should be printed on the console when being run or not.

        :param print_command:   True, if the command line program should be printed, False otherwise
        :return:                The `Command` itself
        """
        self.run_options.print_command = print_command
        return self

    def exit_on_error(self, exit_on_error: bool) -> 'Command':
        """
        Sets whether the build system should be terminated if the program exits with a non-zero exit code or not.

        :param exit_on_error:   True, if the build system should be terminated, False, if a `RuntimeError` should be
                                raised instead
        :return:                The `Command` itself
        """
        self.run_options.exit_on_error = exit_on_error
        return self

    def use_environment(self, environment) -> 'Command':
        """
        Sets the environment to be used for running the command line program.

        :param environment: The environment to be set or None, if the default environment should be used
        :return:            The `Command` itself
        """
        self.run_options.environment = environment
        return self

    def _should_be_skipped(self) -> bool:
        """
        May be overridden by subclasses in order to determine whether the command should be skipped or not.
        """
        return False

    def _before(self):
        """
        May be overridden by subclasses in order to perform some operations before the command is run.
        """

    def run(self):
        """
        Runs the command line program.
        """
        if not self._should_be_skipped():
            self._before()
            self.run_options.run(self, capture_output=False)
            self._after()

    def capture_output(self) -> str:
        """
        Runs the command line program and returns its output.

        :return: The output of the program
        """
        if not self._should_be_skipped():
            self._before()
            output = self.run_options.run(self, capture_output=True)
            self._after()
            return output.stdout

        return ''

    def _after(self):
        """
        May be overridden by subclasses in order to perform some operations after the command has been run.
        """

    def __str__(self) -> str:
        print_options = Command.PrintOptions()
        print_options.print_arguments = True
        return print_options.format(self)
