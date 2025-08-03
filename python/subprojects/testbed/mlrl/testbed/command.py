"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with commands and their arguments.
"""
from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, List, Optional, override


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
        argument_list:  An `ArgumentList` that stores the arguments
    """
    module_name: str
    argument_list: ArgumentList

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
