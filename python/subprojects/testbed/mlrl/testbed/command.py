"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with commands and their arguments.
"""
import sys

from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, override

from mlrl.util.cli import Argument
from mlrl.util.format import format_iterable


class ArgumentList(List[str]):
    """
    A list that stores command line arguments.
    """

    def filter(self, *arguments_to_be_ignored: str) -> 'ArgumentList':
        """
        Creates and returns an `ArgumentList` by filtering the arguments in this list.

        :param arguments_to_be_ignored: The names of the arguments to be ignored
        :return:                        The `ArgumentList` that has been created
        """
        ignored_arguments = set(arguments_to_be_ignored)
        filtered_list = []
        skip = False

        for argument in self:
            skip = (skip and not argument.startswith('-')) or argument in ignored_arguments

            if not skip:
                filtered_list.append(argument)

        return ArgumentList(filtered_list)

    def to_dict(self) -> 'ArgumentDict':
        """
        Creates and returns an `ArgumentDict` from this list.

        :return: The `ArgumentDict` that has been created
        """
        argument_dict: Dict[str, Optional[str]] = {}
        previous_argument = None

        for argument in self:
            if not previous_argument or argument.startswith('-'):
                argument_dict.setdefault(argument, None)
                previous_argument = argument
            else:
                argument_dict[previous_argument] = argument
                previous_argument = None

        return ArgumentDict(argument_dict)


class ArgumentDict(Dict[str, Optional[str]]):
    """
    A dictionary that stores the names of command line arguments, as well as their associated values, if available.
    """

    def to_list(self) -> ArgumentList:
        """
        Creates and returns an `ArgumentList` from this dictionary.

        :return: The `ArgumentList` that has been created
        """
        argument_list = []

        for key in sorted(self.keys()):
            argument_list.append(key)
            value = self.get(key)

            if value:
                argument_list.append(value)

        return ArgumentList(argument_list)


@dataclass
class Command(Iterable[str]):
    """
    A command for running MLRL-Testbed, consisting of a module name and several arguments.

    Attributes:
        module_name:    The name of the Python module
        argument_dict:  An `ArgumentDict` that stores the arguments of the command
        argument_list:  An `ArgumentList` that stores the arguments of the command
    """
    module_name: str
    argument_dict: ArgumentDict
    argument_list: ArgumentList

    @staticmethod
    def from_list(module_name: str, argument_list: ArgumentList) -> 'Command':
        """
        Creates and returns a command from a given `ArgumentList`.

        :param module_name:     The name of the Python module
        :param argument_list:   The `ArgumentList`, the command should be created from
        :return:                The command that has been created
        """
        return Command(module_name=module_name, argument_list=argument_list, argument_dict=argument_list.to_dict())

    @staticmethod
    def from_dict(module_name: str, argument_dict: ArgumentDict) -> 'Command':
        """
        Creates and returns a command from a given `ArgumentDict`.

        :param module_name:     The name of the Python module
        :param argument_dict:   The `ArgumentDict`, the command should be created from
        :return:                The command that has been created
        """
        return Command(module_name=module_name, argument_dict=argument_dict, argument_list=argument_dict.to_list())

    @staticmethod
    def from_string(string: str) -> 'Command':
        """
        Creates and returns a command from a given string.

        :param string:  The string, the command should be created from
        :return:        The command that has been created
        """
        args = string.split()
        return Command.from_list(args[1], ArgumentList(args[2:]))

    @staticmethod
    def from_argv() -> 'Command':
        """
        Creates and returns a command from `sys.argv`.

        :return: The command that has been created
        """
        return Command.from_list(module_name=sys.argv[1], argument_list=ArgumentList(sys.argv[2:]))

    def apply_to_namespace(self, namespace: Namespace, ignore: Optional[Set[str]] = None) -> Namespace:
        """
        Adds the command's arguments to a given namespace.

        :param namespace:   The namespace, the arguments should be added to
        :param ignore:      A set that contains the names of arguments to be ignored or None, if all argument should be
                            added
        :return:            The modified namespace
        """
        argument_list = self.argument_list
        modified_namespace = copy(namespace)
        ignored_argument_names = {Argument.argument_name_to_key(argument) for argument in ignore} if ignore else set()

        for i, argument in enumerate(argument_list):
            if argument.startswith('-'):
                argument_name = Argument.argument_name_to_key(argument)

                if argument_name not in ignored_argument_names:
                    argument_value = None

                    if i + 1 < len(argument_list):
                        next_argument = argument_list[i + 1]

                        if not next_argument.startswith('-'):
                            argument_value = next_argument

                    setattr(modified_namespace, argument_name, argument_value if argument_value else True)

        return modified_namespace

    def to_namespace(self, ignore: Optional[Set[str]] = None) -> Namespace:
        """
        Creates and returns a namespace from the command's arguments.

        :param ignore:  A set that contains the names of arguments to be ignored or None, if all argument should be
                        added
        :return:        The namespace that has been created
        """
        return self.apply_to_namespace(Namespace(), ignore=ignore)

    @staticmethod
    def with_argument(command: 'Command', argument: str, value: str) -> 'Command':
        """
        Creates and returns a modified variant of a command, where a specific argument set to a given value.

        :param command:     The command to modify
        :param argument:    The name of the argument to be set
        :param value:       The value to be set
        :return:            The modified command
        """
        argument_dict = ArgumentDict(command.argument_dict)
        argument_dict[argument] = value
        return Command.from_dict(module_name=command.module_name, argument_dict=argument_dict)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(chain(['mlrl-testbed', self.module_name], self.argument_list))

    @override
    def __str__(self) -> str:
        return format_iterable(self, separator=' ')

    @override
    def __hash__(self) -> int:
        return hash(str(self))

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and str(self) == str(other)
