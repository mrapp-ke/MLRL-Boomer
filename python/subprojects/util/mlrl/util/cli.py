"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for configuring the arguments of a command line interface.
"""
import sys

from argparse import ArgumentError, ArgumentParser, Namespace, _ArgumentGroup
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, override

from mlrl.util.format import format_enum_values, format_set, format_value
from mlrl.util.options import BooleanOption, Options, parse_enum, parse_param, parse_param_and_options
from mlrl.util.validation import ValidationError

NONE = 'none'

AUTO = 'auto'


class Argument:
    """
    A single argument of a command line interface for which the user can provide a custom value.
    """

    def __init__(self,
                 *names: str,
                 required: bool = False,
                 default: Optional[Any] = None,
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 **kwargs):
        """
        :param names:                               One or several names of the argument
        :param required:                            True, if the argument is mandatory, False otherwise
        :param default:                             The default value of the argument, if any
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param kwargs:                              Optional keyword argument to be passed to an `ArgumentParser`
        """
        self.names = set(names)
        self.required = required
        self.default = default

        if description is not None:
            if not description.endswith('.'):
                description += '.'

            if add_default_value_to_description and not required and default is not None:
                description += ' The default value is ' + format_value(default) + '.'

        self.description = description
        self.kwargs = dict(kwargs)

    @staticmethod
    def argument_name_to_key(name: str) -> str:
        """
        Converts the name of an argument into its key.

        :param name:    The name of the argument
        :return:        The key of the argument
        """
        return name.lstrip('-').replace('-', '_')

    @cached_property
    def name(self) -> str:
        """
        The name of the argument.
        """
        return sorted(self.names)[0]

    @cached_property
    def key(self) -> str:
        """
        The key of the argument in a `Namespace`.
        """
        return self.argument_name_to_key(self.name)

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

    def get_value_and_options(self, args: Namespace, default: Optional[Any] = None) -> Tuple[Optional[Any], Options]:
        """
        Returns the value provided by the user for this argument.

        :param args:    A `Namespace` that provides access to the values provided by the user
        :param default: The default value to be returned if no value is available
        :return:        The value provided by the user or `default`, if no value is available
        """
        value = self.get_value(args, default=default)

        if value is None:
            return None, Options()

        try:
            unpacked_value, options = value
            return unpacked_value, options
        except TypeError:
            return value, Options()

    @override
    def __hash__(self) -> int:
        return hash(self.key)

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.key == other.key

    @override
    def __str__(self) -> str:
        return self.name


class FlagArgument(Argument):
    """
    An argument of a command line interface, which can be set by the user as a flag.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """
        :param name:        The name of the argument
        :param description: An optional description of the argument
        """
        super().__init__(name,
                         default=None,
                         description=description,
                         action='store_true',
                         add_default_value_to_description=False)


class StringArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom string value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 default: Optional[str] = None,
                 required: bool = False):
        """
        :param names:                               One or several names of the argument
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param default:                             The default value
        :param required:                            True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names,
                         default=default,
                         description=description,
                         add_default_value_to_description=add_default_value_to_description,
                         type=str,
                         required=required)

    @override
    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)
        return None if value is None else str(value)


class PathArgument(StringArgument):
    """
    An argument of a command line interface for which the user can provide a custom string value.
    """

    @override
    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)
        return None if value is None else Path(value)


class IntArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom integer value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 default: Optional[int] = None,
                 required: bool = False):
        """
        :param names:                               One or several names of the argument
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param default:                             The default value
        :param required:                            True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names,
                         default=default,
                         description=description,
                         add_default_value_to_description=add_default_value_to_description,
                         type=int,
                         required=required)

    @override
    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)

        try:
            return None if value is None else int(value)
        except ValueError as error:
            raise ValidationError('Expected value of argument ' + self.name + ' to be an integer, but got: '
                                  + str(value)) from error


class FloatArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom floating point value.
    """

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 default: Optional[float] = None,
                 required: bool = False):
        """
        :param names:                               One or several names of the argument
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param default:                             The default value
        :param required:                            True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names,
                         default=default,
                         description=description,
                         add_default_value_to_description=add_default_value_to_description,
                         type=float,
                         required=required)

    @override
    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)

        try:
            return None if value is None else float(value)
        except ValueError as error:
            raise ValidationError('Expected value of argument ' + self.name + ' to be a float, but got: '
                                  + str(value)) from error


class BoolArgument(Argument):
    """
    An argument of a command line interface for which the user can provide a custom boolean value.
    """

    @staticmethod
    def __format_description(description: Optional[str],
                             has_options: bool,
                             default: Optional[bool] = None,
                             add_default_value_to_description: bool = True) -> str:
        if description:
            if not description.endswith('.'):
                description += '.'

            description += ' '
        else:
            description = ''

        description += 'Must be one of ' + format_enum_values(BooleanOption) + '.'

        if has_options:
            description += ' For additional options refer to the documentation.'

        if add_default_value_to_description:
            description += ' The default value is ' + format_value(
                BooleanOption.TRUE if default else BooleanOption.FALSE) + '.'

        return description

    def __init__(self,
                 *names: str,
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 default: Optional[bool] = None,
                 required: bool = False,
                 true_options: Optional[Set[str]] = None,
                 false_options: Optional[Set[str]] = None):
        """
        :param names:                               One or several names of the argument
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param default:                             The default value
        :param required:                            True, if the argument is mandatory, False otherwise
        :param true_options:                        The names of options that can be provided by the user in addition to
                                                    the value "true"
        :param false_options:                       The names of options that can be provided by the user in addition to
                                                    the value "false"
        """
        super().__init__(*names,
                         default=None if default is None else (BooleanOption.TRUE if default else BooleanOption.FALSE),
                         description=self.__format_description(
                             description,
                             has_options=bool(true_options) or bool(false_options),
                             default=default,
                             add_default_value_to_description=add_default_value_to_description),
                         add_default_value_to_description=False,
                         type=str if true_options or false_options else BooleanOption.parse,
                         required=required)
        self.true_options = true_options if true_options else set()
        self.false_options = false_options if false_options else set()

    @override
    def get_value(self, args: Namespace, default: Optional[Any] = None) -> Optional[Any]:
        value = super().get_value(args, default=default)
        true_options = self.true_options
        false_options = self.false_options
        has_options = bool(true_options) or bool(false_options)

        if value is not None:
            str_value = str(value).lower()

            if has_options:
                str_value, options = parse_param_and_options(self.key, str_value, {
                    str(BooleanOption.TRUE): true_options,
                    str(BooleanOption.FALSE): false_options
                })
                return BooleanOption.parse(str_value), options

            return BooleanOption.parse(str_value)

        return (None, None) if has_options else None


class SetArgument(Argument):
    """
    An argument of a command line interface for which the user can provide one out of a predefined set of string values.
    """

    @staticmethod
    def __format_description(description: Optional[str], values: Set[str] | Dict[str, Set[str]]) -> str:
        if description:
            if not description.endswith('.'):
                description += '.'

            description += ' '
        else:
            description = ''

        description += 'Must be one of ' + format_set(values.keys() if isinstance(values, dict) else values) + '.'

        if isinstance(values, dict):
            description += ' For additional options refer to the documentation.'

        return description

    def __init__(self,
                 *names: str,
                 values: Set[str] | Dict[str, Set[str]],
                 description: Optional[str] = None,
                 add_default_value_to_description: bool = True,
                 default: Optional[str] = None,
                 required: bool = False):
        """
        :param names:                               One or several names of the argument
        :param values:                              A set that contains the predefined values or a dictionary that
                                                    contains the predefined values, as well as the names of options that
                                                    can be provided by the user in addition to the respective values
        :param description:                         An optional description of the argument
        :param add_default_value_to_description:    True, if the default value should be added to the description, if it
                                                    is not None, False otherwise
        :param default:                             The default value
        :param required:                            True, if the argument is mandatory, False otherwise
        """
        super().__init__(*names,
                         default=default,
                         description=self.__format_description(description, values),
                         add_default_value_to_description=add_default_value_to_description,
                         type=str,
                         required=required)
        self.supported_values = values

    @override
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
                 enum: Type[Enum],
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

    @override
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
        self.arguments: List[Argument] = []
        self._argument_parser = argument_parser
        self._argument_groups: Dict[str, _ArgumentGroup] = {}

        if version_text:
            argument_parser.add_argument('-v',
                                         '--version',
                                         action='version',
                                         version=version_text,
                                         help='Display information about the program.')

    def add_arguments(self, *arguments: Argument, group: Optional[str] = None):
        """
        Adds a new argument that enables the user to provide a value to the command line interface.

        :param arguments:   The arguments to be added
        :param group:       The name of a group, the arguments should be added to, or None, if they should not be added
                            to a particular group
        """
        argument_parser = self._argument_parser
        argument_group = self._argument_groups.setdefault(
            group, argument_parser.add_argument_group(group)) if group else argument_parser

        for argument in arguments:
            try:
                required = argument.required and '--help' not in sys.argv and '-h' not in sys.argv
                argument_group.add_argument(*argument.names,
                                            required=required,
                                            default=argument.default,
                                            help=argument.description,
                                            **argument.kwargs)
                self.arguments.append(argument)
            except ArgumentError:
                # Argument has already been added
                pass

    def parse_known_args(self) -> Namespace:
        """
        Parses and returns the values of the arguments already added to the command line interface.

        :return: A `Namespace` providing access to the values of the arguments
        """
        return self._argument_parser.parse_known_args()[0]
