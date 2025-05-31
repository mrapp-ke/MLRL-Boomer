"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for configuring a command line interface.
"""
from argparse import ArgumentParser
from typing import Optional

from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.cli import Argument


class CommandLineInterface:
    """
    Allows to configure a command line interface for running a program.
    """

    def __init__(self, argument_parser: ArgumentParser, program_info: Optional[ProgramInfo] = None):
        """
        :param argument_parser: The parser that should be used for parsing arguments provided to the command line
                                interface by the user
        :param program_info:    Information about the program to be shown when the "--version" flag is passed to the
                                command line interface
        """
        self.argument_parser = argument_parser

        if program_info:
            argument_parser.add_argument('-v',
                                         '--version',
                                         action='version',
                                         version=str(program_info),
                                         help='Display information about the program.')

    def add_arguments(self, *arguments: Argument) -> 'CommandLineInterface':
        """
        Adds a new argument that enables the user to provide a value to the command line interface.

        :param arguments:   The arguments to be added
        :return:            The command line interface itself
        """
        argument_parser = self.argument_parser

        for argument in arguments:
            argument.add_to_argument_parser(argument_parser)

        return self
