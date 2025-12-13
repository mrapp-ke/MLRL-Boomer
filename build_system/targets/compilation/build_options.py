"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to configure build options.
"""
from abc import ABC, abstractmethod
from os import environ
from typing import Any, Iterable, List, Optional, override

from util.env import get_env, get_env_array

from targets.modules import SubprojectModule
from targets.project import Project


class BuildOption(ABC):
    """
    An abstract base class for all build options.
    """

    def __init__(self, name: str, *subprojects: str):
        """
        :param name:        The name of the build option
        :param subprojects: The subprojects, the build option applies to, or no subproject, if it is a global option
        """
        self.name = name
        self.subprojects = list(subprojects)

    @property
    def keys(self) -> List[str]:
        """
        A list of keys to be used for setting the build option.
        """
        subprojects = self.subprojects

        if subprojects:
            subproject_names = set(get_env_array(environ, SubprojectModule.ENV_SUBPROJECTS))

            if not subproject_names or any(subproject in subproject_names for subproject in subprojects):
                return [subproject + ':' + self.name for subproject in subprojects]

            cpp_subprojects = {subproject.name for subproject in Project.Cpp.find_subprojects()}

            if any(subproject_name in cpp_subprojects for subproject_name in subproject_names):
                return [subproject + ':' + self.name for subproject in subprojects if subproject == 'common']

        return [self.name]

    @property
    @abstractmethod
    def value(self) -> Optional[str]:
        """
        Returns the value of the build option.

        :return: The value or None, if no value is set
        """

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.keys == other.keys

    @override
    def __hash__(self) -> int:
        return hash(tuple(self.keys))

    def __bool__(self) -> bool:
        return self.value is not None


class ConstantBuildOption(BuildOption):
    """
    A build options with a hardcoded value.
    """

    def __init__(self, name: str, value: str, *subprojects: str):
        """
        :param name:        The name of the build option
        :param value:       The value of the build option
        :param subprojects: The subprojects, the build option applies to, or no subproject, if it is a global option
        """
        super().__init__(name, *subprojects)
        self._value = value

    @override
    @property
    def value(self) -> Optional[str]:
        return self._value


class EnvBuildOption(BuildOption):
    """
    A build option, whose value is obtained from an environment variable.
    """

    def __init__(self, name: str, *subprojects: str, default_value: Optional[str] = None):
        """
        :param name:            The name of the build option
        :param subprojects:     The subprojects, the build option applies to, or no subproject, if it is a global option
        :param default_value:   An optional default value
        """
        super().__init__(name, *subprojects)
        self.default_value = default_value

    @override
    @property
    def value(self) -> Optional[str]:
        value = get_env(environ, self.name.upper(), self.default_value)

        if value:
            value = value.strip()

        return value


class BuildOptions(Iterable[BuildOption]):
    """
    Stores multiple build options.
    """

    def __init__(self):
        self.build_options = set()

    def add(self, build_option: BuildOption) -> 'BuildOptions':
        """
        Adds a build option.

        :param build_option:    The build option to be added
        :return:                The `BuildOptions` itself
        """
        self.build_options.add(build_option)
        return self

    @override
    def __iter__(self):
        return iter(self.build_options)

    def __bool__(self) -> bool:
        for build_option in self.build_options:
            if build_option:
                return True

        return False
