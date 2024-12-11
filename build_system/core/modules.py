"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about files and directories that belong to individual modules of the project
to be dealt with by the targets of the build system.
"""
from abc import ABC
from functools import reduce
from typing import List


class Module(ABC):
    """
    An abstract base class for all modules.
    """

    class Filter(ABC):
        """
        An abstract base class for all classes that allow to filter modules.
        """

        def matches(self, module: 'Module') -> bool:
            """
            Returns whether the filter matches a given module or not.

            :param module:  The module to be matched
            :return:        True, if the filter matches the given module, False otherwise
            """

    def match(self, module_filter: Filter) -> List['Module']:
        """
        Returns a list that contains all submodules in this module that match a given filter.

        :param module_filter:   The filter
        :return:                A list that contains all matching submodules
        """
        return [self] if module_filter.matches(self) else []


class ModuleRegistry:
    """
    Allows to look up modules that have previously been registered.
    """

    def __init__(self):
        self.modules = []

    def register(self, module: Module):
        """
        Registers a new module.

        :param module: The module to be registered
        """
        self.modules.append(module)

    def lookup(self, module_filter: Module.Filter) -> List[Module]:
        """
        Looks up and returns all modules that match a given filter.

        :param module_filter:   The filter
        :return:                A list that contains all modules matching the given filter
        """
        return list(reduce(lambda aggr, module: aggr + module.match(module_filter), self.modules, []))
