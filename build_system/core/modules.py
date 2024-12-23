"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about files and directories that belong to individual modules of the project
to be dealt with by the targets of the build system.
"""
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional, Set

from util.env import get_env_array


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


class SubprojectModule(Module, ABC):
    """
    An abstract base class for all modules that correspond to one of several subprojects.
    """

    ENV_SUBPROJECTS = 'SUBPROJECTS'

    class Filter(Module.Filter):
        """
        An abstract base class for all classes that allow to filter modules by subprojects.
        """

        def __init__(self, subproject_names: Optional[Set[str]] = None):
            """
            :param subproject_names: A set that contains the names of the subprojects to be matched or None, if no
                                     restrictions should be imposed
            """
            self.subproject_names = subproject_names

        @staticmethod
        def from_env(env: Dict) -> 'SubprojectModule.Filter':
            """
            Creates and returns a `SubprojectModule.Filter` that filters modules by the subprojects given via the
            environment variable `SubprojectModule.ENV_SUBPROJECTS`.

            :param env: The environment to be accessed
            :return:    The `SubprojectModule.Filter` that has been created
            """
            return SubprojectModule.Filter(set(get_env_array(env, SubprojectModule.ENV_SUBPROJECTS)))

        def matches(self, module: 'Module') -> bool:
            """
            Returns whether the filter matches a given module or not.

            :param module:  The module to be matched
            :return:        True, if the filter matches the given module, False otherwise
            """
            return isinstance(module, SubprojectModule) and (not self.subproject_names
                                                             or module.subproject_name in self.subproject_names)

    @property
    @abstractmethod
    def subproject_name(self) -> str:
        """
        The name of the subproject, the module corresponds to.
        """


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
