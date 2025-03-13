"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about files and directories that belong to individual modules of the project
to be dealt with by the targets of the build system.
"""
from abc import ABC
from typing import List


class Module(ABC):
    """
    An abstract base class for all modules.
    """

    class Filter(ABC):
        """
        An abstract base class for all classes that allow to filter modules.
        """

        def matches(self, module: 'Module', module_registry: 'ModuleRegistry') -> bool:
            """
            Returns whether the filter matches a given module or not.

            :param module:          The module to be matched
            :param module_registry: A `ModuleRegistry` that allows to look up any registered modules
            :return:                True, if the filter matches the given module, False otherwise
            """

    def match(self, module_filter: Filter, module_registry: 'ModuleRegistry') -> List['Module']:
        """
        Returns a list that contains all submodules in this module that match a given filter.

        :param module_filter:   The filter
        :param module_registry: A `ModuleRegistry` that allows to look up any registered modules
        :return:                A list that contains all matching submodules
        """
        return [self] if module_filter.matches(self, module_registry) else []


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

    def lookup(self, module_filter: Module.Filter, *additional_filters: Module.Filter) -> List[Module]:
        """
        Looks up and returns all modules that match one or several filters.

        :param module_filter:       The filter
        :param additional_filters:  Additional filters that must also be matched
        :return:                    A list that contains all modules matching the given filters
        """
        matched_modules = []

        for module in self.modules:
            if module.match(module_filter, self) and all(
                    module.match(additional_filter, self) for additional_filter in additional_filters):
                matched_modules.append(module)

        return matched_modules
