"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for configuring a command line interface.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.format import format_enum_values, format_set
from mlrl.util.options import BooleanOption


@dataclass
class Argument:
    """
    A single argument of a command line interface for which the user can provide a custom value.

    Attributes:
        names:      One of several names of the argument
        help:       An optional description of the argument
        type:       The type of the value
        default:    The default value
        required:   True, if the argument is mandatory, False otherwise
    """
    names: List[str]
    help: Optional[str] = None
    type: Type = str
    default: Optional[Any] = None
    required: bool = False

    @staticmethod
    def string(*names: str,
               help: Optional[str] = None,
               default: Optional[str] = None,
               required: bool = False) -> 'Argument':
        """
        Creates and returns a new argument for which the user can provide a custom string value.

        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        :return:            The argument that has been created
        """
        return Argument(names=list(names), help=help, type=str, default=default, required=required)

    @staticmethod
    def int(*names: str,
            help: Optional[str] = None,
            default: Optional[int] = None,
            required: bool = False) -> 'Argument':
        """
        Creates and returns a new argument for which the user can provide a custom integer value.

        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        :return:            The argument that has been created
        """
        return Argument(names=list(names), help=help, type=int, default=default, required=required)

    @staticmethod
    def float(*names: str,
              help: Optional[str] = None,
              default: Optional[int] = None,
              required: bool = False) -> 'Argument':
        """
        Creates and returns a new argument for which the user can provide a custom floating point value.

        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        :return:            The argument that has been created
        """
        return Argument(names=list(names), help=help, type=float, default=default, required=required)

    @staticmethod
    def bool(*names: str,
             help: Optional[str] = None,
             default: bool = False,
             required: bool = False,
             true_options: Optional[Set[str]] = None,
             false_options: Optional[Set[str]] = None) -> 'Argument':
        """
        Creates and returns a new argument for which the user can provide a custom boolean value.

        :param names:           One or several names of the argument
        :param help:            An optional description of the argument
        :param default:         The default value
        :param required:        True, if the argument is mandatory, False otherwise
        :param true_options:    The names of options that can be provided by the user in addition to the value "true"
        :param false_options:   The names of options that can be provided by the user in addition to the value "false"
        :return:                The argument that has been created
        """
        if not help.endswith('.'):
            help += '.'

        help += ' Must be one of ' + format_enum_values(BooleanOption) + '.'
        has_options = true_options or false_options

        if has_options:
            help += ' For additional options refer to the documentation.'
            type = str
            default = None if default is None else (BooleanOption.TRUE.value if default else BooleanOption.FALSE.value)
        else:
            type = BooleanOption.parse

        return Argument(names=list(names), help=help, type=type, default=default, required=required)

    @staticmethod
    def set(*names: str,
            values: Enum | Set[str] | Dict[str, Set[str]],
            help: Optional[str] = None,
            default: Optional[str] = None,
            required: bool = False) -> 'Argument':
        """
        Creates and returns a new argument for which the user can provide one out of a predefined set of string values.

        :param names:       One or several names of the argument
        :param values:      An enum or set that contains the predefined values or a dictionary that contains the
                            predefined values, as well as the names of options that can be provided by the user in
                            addition to the respective values
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        :return:            The argument that has been created
                """
        if not help.endswith('.'):
            help += '.'

        help += ' Must be one of '

        if isinstance(values, Enum):
            help += format_enum_values(values)
        else:
            help += format_set(values.keys() if isinstance(values, dict) else values)

        help += '.'

        if isinstance(values, dict):
            help += ' For additional options refer to the documentation.'

        return Argument(names=list(names), help=help, type=str, default=default, required=required)


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
        for argument in arguments:
            self.argument_parser.add_argument(*argument.names,
                                              type=argument.type,
                                              default=argument.default,
                                              help=argument.help,
                                              required=argument.required)

        return self
