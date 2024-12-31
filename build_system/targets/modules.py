"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about files and directories that belong to individual modules of the project
to be dealt with by the targets of the build system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from core.modules import Module
from util.env import get_env_array


class SubprojectModule(Module, ABC):
    """
    An abstract base class for all modules that correspond to one of several subprojects.
    """

    ENV_SUBPROJECTS = 'SUBPROJECTS'

    SUBPROJECT_COMMON = 'common'

    SUBPROJECT_TESTBED = 'testbed'

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
        def from_env(env: Dict, always_match: Optional[Set[str]] = None) -> 'SubprojectModule.Filter':
            """
            Creates and returns a `SubprojectModule.Filter` that filters modules by the subprojects given via the
            environment variable `SubprojectModule.ENV_SUBPROJECTS`.

            :param env:             The environment to be accessed
            :param always_match:    A set that contains the names of the subprojects that should always be matched by
                                    the filter
            :return:                The `SubprojectModule.Filter` that has been created
            """
            subproject_names = set(get_env_array(env, SubprojectModule.ENV_SUBPROJECTS))

            if subproject_names and always_match:
                subproject_names.update(always_match)

            return SubprojectModule.Filter(subproject_names)

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
