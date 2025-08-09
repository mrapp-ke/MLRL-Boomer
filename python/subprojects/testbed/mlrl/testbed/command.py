"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with commands and their arguments.
"""
import sys

from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, List, Optional, override

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
    def from_argv() -> 'Command':
        """
        Creates and returns a command from `sys.argv`.

        :return: The command that has been created
        """
        return Command.from_list(module_name=sys.argv[1], argument_list=ArgumentList(sys.argv[2:]))

    def apply_to_namespace(self, namespace: Namespace) -> Namespace:
        """
        Adds the command's arguments to a given namespace.

        :param namespace:   The namespace, the arguments should be added to
        :return:            The modified namespace
        """
        argument_list = self.argument_list
        modified_namespace = copy(namespace)

        for i, argument in enumerate(argument_list):
            if argument.startswith('-'):
                argument_name = argument.lstrip('-').replace('-', '_')
                argument_value = None

                if i + 1 < len(argument_list):
                    next_argument = argument_list[i + 1]

                    if not next_argument.startswith('-'):
                        argument_value = next_argument

                setattr(modified_namespace, argument_name, argument_value if argument_value else True)

        return modified_namespace

    @override
    def __iter__(self):
        return iter(chain(['mlrl-testbed', self.module_name], self.argument_list))

    @override
    def __str__(self) -> str:
        return format_iterable(self, separator=' ')
