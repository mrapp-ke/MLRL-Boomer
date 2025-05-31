"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for configuring the arguments of a command line interface.
"""
from argparse import ArgumentError, ArgumentParser, Namespace
from enum import Enum
from typing import Any, Dict, Optional, Set

from mlrl.util.format import format_enum_values, format_set
from mlrl.util.options import BooleanOption


class Argument:
    """
    An abstract base class for all arguments of a command line interface for which the user can provide a custom value.
    """

    def __init__(self, *names: str, **kwargs: Any):
        """
        :param names:   One of several names of the argument
        :param kwargs:  Optional keyword argument to be passed to an `ArgumentParser`
        """
        self.names = set(names)
        self.kwargs = dict(kwargs)


class StringArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom string value.
    """

    def __init__(self, *names: str, help: Optional[str] = None, default: Optional[str] = None, required: bool = False):
        """
        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, help=help, type=str, default=default, required=required)


class IntArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom integer value.
    """

    def __init__(self, *names: str, help: Optional[str] = None, default: Optional[int] = None, required: bool = False):
        """
        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, help=help, type=int, default=default, required=required)


class FloatArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom floating point value.
    """

    def __init__(self, *names: str, help: Optional[str] = None, default: Optional[int] = None, required: bool = False):
        """
        :param names:       One or several names of the argument
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, help=help, type=float, default=default, required=required)


class BoolArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom boolean value.
    """

    @staticmethod
    def __format_help(help: str, has_options: bool) -> str:
        if not help.endswith('.'):
            help += '.'

        help += ' Must be one of ' + format_enum_values(BooleanOption) + '.'

        if has_options:
            help += ' For additional options refer to the documentation.'

        return help

    def __init__(self,
                 *names: str,
                 help: Optional[str] = None,
                 default: bool = False,
                 required: bool = False,
                 true_options: Optional[Set[str]] = None,
                 false_options: Optional[Set[str]] = None):
        """
        :param names:           One or several names of the argument
        :param help:            An optional description of the argument
        :param default:         The default value
        :param required:        True, if the argument is mandatory, False otherwise
        :param true_options:    The names of options that can be provided by the user in addition to the value "true"
        :param false_options:   The names of options that can be provided by the user in addition to the value "false"
        """
        super().__init__(*names,
                         help=self.__format_help(help,
                                                 bool(true_options) or bool(false_options)),
                         type=str if true_options or false_options else BooleanOption.parse,
                         default=None if default is None else
                         (BooleanOption.TRUE.value if default else BooleanOption.FALSE.value),
                         required=required)


class SetArgument(Argument):
    """
    An argument of a command line interface for which the user can provide one out of a predefined set of string values.
    """

    @staticmethod
    def __format_help(help: str, values: Any) -> str:
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

        return help

    def __init__(self,
                 *names: str,
                 values: Enum | Set[str] | Dict[str, Set[str]],
                 help: Optional[str] = None,
                 default: Optional[str] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param values:      An enum or set that contains the predefined values or a dictionary that contains the
                            predefined values, as well as the names of options that can be provided by the user in
                            addition to the respective values
        :param help:        An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, help=self.__format_help(help, values), type=str, default=default, required=required)


class CommandLineInterface:
    """
    Allows to configure a command line interface for running a program.
    """

    def __init__(self, argument_parser: ArgumentParser, version_text: Optional[str] = None):
        """
        :param argument_parser: The parser that should be used for parsing arguments provided to the command line
                                interface by the user
        :param version_text:    A text to be shown when the "--version" flag is passed to the command line interface or
                                None, if the "--version" flag should not be added to the command line interface
        """
        self._argument_parser = argument_parser

        if version_text:
            argument_parser.add_argument('-v',
                                         '--version',
                                         action='version',
                                         version=version_text,
                                         help='Display information about the program.')

    def add_arguments(self, *arguments: Argument, return_known_args: bool = False) -> Optional[Namespace]:
        """
        Adds a new argument that enables the user to provide a value to the command line interface.

        :param arguments:           The arguments to be added
        :param return_known_args:   True, if the values of the arguments already added to the command line interface
                                    should be parsed and returned, False otherwise
        :return:                    A `Namespace` providing access to the values of the arguments already added or None
        """
        argument_parser = self._argument_parser

        for argument in arguments:
            try:
                argument_parser.add_argument(*argument.names, **argument.kwargs)
            except ArgumentError:
                # Argument has already been added
                pass

        return argument_parser.parse_known_args()[0] if return_known_args else None
