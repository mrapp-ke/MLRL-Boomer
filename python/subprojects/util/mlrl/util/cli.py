"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for configuring the arguments of a command line interface.
"""
import sys

from argparse import ArgumentError, ArgumentParser, Namespace
from enum import Enum, EnumType
from functools import cached_property
from typing import Any, Dict, Optional, Set

from mlrl.util.format import format_enum_values, format_set
from mlrl.util.options import BooleanOption, parse_enum, parse_param, parse_param_and_options

NONE = 'none'


class Argument:
    """
    A single argument of a command line interface for which the user can provide a custom value.
    """

    def __init__(self, *names: str, required: Optional[bool] = False, default: Optional[Any] = None, **kwargs: Any):
        """
        :param names:       One of several names of the argument
        :param required:    True, if the argument is mandatory, False otherwise
        :param default:     The default value of the argument, if any
        :param kwargs:      Optional keyword argument to be passed to an `ArgumentParser`
        """
        self.names = set(names)
        self.required = required
        self.default = default
        self.kwargs = dict(kwargs)

    @cached_property
    def name(self) -> str:
        """
        The name of the argument.
        """
        return next(iter(self.names))

    @cached_property
    def key(self) -> str:
        """
        The key of the argument in a `Namespace`.
        """
        return self.name.lstrip('--').replace('-', '_')

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        """
        Returns the value provided by the user for this argument.

        :param args:    A `Namespace` that provides access to the values provided by the user
        :param default: The default value to be returned if no value is available
        :return:        The value provided by the user or `default`, if no value is available
        """
        value = getattr(args, self.key, None)
        value = self.default if value is None else value
        return default if value is None else value

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.key == other.key


class StringArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom string value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 default: Optional[str] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param description: An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, default=default, help=description, type=str, required=required)

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)
        return None if value is None else str(value)


class IntArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom integer value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 default: Optional[int] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param description: An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, default=default, help=description, type=int, required=required)

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)

        try:
            return None if value is None else int(value)
        except ValueError as error:
            raise ValueError('Expected value of argument ' + self.name + ' to be an integer, but got: '
                             + str(value)) from error


class FloatArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom floating point value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 default: Optional[float] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param description: An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names, default=default, help=description, type=float, required=required)

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)

        try:
            return None if value is None else float(value)
        except ValueError as error:
            raise ValueError('Expected value of argument ' + self.name + ' to be a float, but got: '
                             + str(value)) from error


class BoolArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom boolean value.
    """

    @staticmethod
    def __format_description(description: str, has_options: bool) -> str:
        if not description.endswith('.'):
            description += '.'

        description += ' Must be one of ' + format_enum_values(BooleanOption) + '.'

        if has_options:
            description += ' For additional options refer to the documentation.'

        return description

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 default: Optional[bool] = None,
                 required: bool = False,
                 true_options: Optional[Set[str]] = None,
                 false_options: Optional[Set[str]] = None):
        """
        :param names:           One or several names of the argument
        :param description:     An optional description of the argument
        :param default:         The default value
        :param required:        True, if the argument is mandatory, False otherwise
        :param true_options:    The names of options that can be provided by the user in addition to the value "true"
        :param false_options:   The names of options that can be provided by the user in addition to the value "false"
        """
        super().__init__(*names,
                         default=None if default is None else (BooleanOption.TRUE if default else BooleanOption.FALSE),
                         help=self.__format_description(description,
                                                        bool(true_options) or bool(false_options)),
                         type=str if true_options or false_options else BooleanOption.parse,
                         required=required)
        self.true_options = true_options
        self.false_options = false_options

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = str(super().get_value(args, default=default)).lower()

        if value:
            true_options = self.true_options
            false_options = self.false_options

            if true_options or false_options:
                value, options = parse_param_and_options(self.key, value, {
                    BooleanOption.TRUE: true_options,
                    BooleanOption.FALSE: false_options
                })
                return BooleanOption.parse(value), options

            return BooleanOption.parse(value)

        return None


class SetArgument(Argument):
    """
    An argument of a command line interface for which the user can provide one out of a predefined set of string values.
    """

    @staticmethod
    def __format_description(description: str, values: Set[str] | Dict[str, Set[str]]) -> str:
        if not description.endswith('.'):
            description += '.'

        description += ' Must be one of ' + format_set(values.keys() if isinstance(values, dict) else values) + '.'

        if isinstance(values, dict):
            description += ' For additional options refer to the documentation.'

        return description

    def __init__(self,
                 *names: str,
                 values: Set[str] | Dict[str, Set[str]],
                 description: Optional[str] = None,
                 default: Optional[str] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param values:      A set that contains the predefined values or a dictionary that contains the predefined
                            values, as well as the names of options that can be provided by the user in addition to the
                            respective values
        :param description: An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names,
                         default=default,
                         help=self.__format_description(description, values),
                         type=str,
                         required=required)
        self.supported_values = values

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)

        if value:
            supported_values = self.supported_values

            if isinstance(supported_values, dict):
                return parse_param_and_options(self.key, value, supported_values)

            return parse_param(self.key, value, supported_values)

        return None


class EnumArgument(SetArgument):
    """
    An argument of a command line interface for which the user can provide one out of a predefined set enum values.
    """

    def __init__(self,
                 *names: str,
                 enum: EnumType,
                 description: Optional[str] = None,
                 default: Optional[Enum] = None,
                 required: bool = False):
        """
        :param names:       One or several names of the argument
        :param values:      An enum that contains the predefined values
        :param description: An optional description of the argument
        :param default:     The default value
        :param required:    True, if the argument is mandatory, False otherwise
        """
        super().__init__(
            *names,
            values={x.value if isinstance(x.value, str) else x.name.lower()
                    for x in enum},
            description=description,
            default=(default.value if isinstance(default.value, str) else default.name.lower()) if default else None,
            required=required)
        self.enum = enum

    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)
        return parse_enum(self.name, value, self.enum) if value else None


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

    def add_arguments(self, *arguments: Argument) -> Optional[Namespace]:
        """
        Adds a new argument that enables the user to provide a value to the command line interface.

        :param arguments: The arguments to be added
        """
        argument_parser = self._argument_parser

        for argument in arguments:
            try:
                required = argument.required and '--help' not in sys.argv and '-h' not in sys.argv
                argument_parser.add_argument(*argument.names,
                                             required=required,
                                             default=argument.default,
                                             **argument.kwargs)
            except ArgumentError:
                # Argument has already been added
                pass

    def parse_known_args(self) -> Namespace:
        """
        Parses and returns the values of the arguments already added to the command line interface.

        :return: A `Namespace` providing access to the values of the arguments
        """
        return self._argument_parser.parse_known_args()[0]
