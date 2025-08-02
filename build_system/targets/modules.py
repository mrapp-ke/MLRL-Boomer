"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about files and directories that belong to individual modules of the project
to be dealt with by the targets of the build system.
"""
from abc import ABC, abstractmethod
from typing import Optional, Set, override

from core.modules import Module, ModuleRegistry
from util.env import Env, get_env_array


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
        def from_env(env: Env) -> 'SubprojectModule.Filter':
            """
            Creates and returns a `SubprojectModule.Filter` that filters modules by the subprojects given via the
            environment variable `SubprojectModule.ENV_SUBPROJECTS`.

            :param env: The environment to be accessed
            :return:    The `SubprojectModule.Filter` that has been created
            """
            subproject_names = set(get_env_array(env, SubprojectModule.ENV_SUBPROJECTS))
            return SubprojectModule.Filter(subproject_names)

        # pylint: disable=unused-argument
        @override
        def matches(self, module: Module, module_registry: ModuleRegistry) -> bool:
            """
            Returns whether the filter matches a given module or not.

            :param module:          The module to be matched
            :param module_registry: A `ModuleRegistry` that allows to look up any registered modules
            :return:                True, if the filter matches the given module, False otherwise
            """
            return isinstance(module, SubprojectModule) and (not self.subproject_names
                                                             or module.subproject_name in self.subproject_names)

    @property
    @abstractmethod
    def subproject_name(self) -> str:
        """
        The name of the subproject, the module corresponds to.
        """
