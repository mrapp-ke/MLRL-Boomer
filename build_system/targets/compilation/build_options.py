"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to configure build options.
"""
from abc import ABC, abstractmethod
from os import environ
from typing import Iterable, Optional

from util.env import get_env


class BuildOption(ABC):
    """
    An abstract base class for all build options.
    """

    def __init__(self, name: str, subproject: Optional[str] = None):
        """
        :param name:        The name of the build option
        :param subproject:  The subproject, the build option corresponds to, or None, if it is a global option
        """
        self.name = name
        self.subproject = subproject

    @property
    def key(self) -> str:
        """
        The key to be used for setting the build option.
        """
        return (self.subproject + ':' if self.subproject else '') + self.name

    @property
    @abstractmethod
    def value(self) -> Optional[str]:
        """
        Returns the value of the build option.

        :return: The value or None, if no value is set
        """

    def __eq__(self, other: 'BuildOption') -> bool:
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __bool__(self) -> bool:
        return self.value is not None


class ConstantBuildOption(BuildOption):
    """
    A build options with a hardcoded value.
    """

    def __init__(self, name: str, value: str, subproject: Optional[str] = None):
        """
        :param name:        The name of the build option
        :param value:       The value of the build option
        :param subproject:  The subpackage, the build option corresponds to, or None, if it is a global option
        """
        super().__init__(name, subproject)
        self._value = value

    @property
    def value(self) -> Optional[str]:
        return self._value


class EnvBuildOption(BuildOption):
    """
    A build option, whose value is obtained from an environment variable.
    """

    def __init__(self, name: str, default_value: Optional[str] = None, subproject: Optional[str] = None):
        """
        :param name:            The name of the build option
        :param default_value:   An optional default value
        :param subproject:      The subproject, the build option corresponds to, or None, if it is a global option
        """
        super().__init__(name, subproject)
        self.default_value = default_value

    @property
    def value(self) -> Optional[str]:
        value = get_env(environ, self.name.upper(), self.default_value)

        if value:
            value = value.strip()

        return value


class BuildOptions(Iterable):
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

    def __iter__(self):
        return iter(self.build_options)

    def __bool__(self) -> bool:
        for build_option in self.build_options:
            if build_option:
                return True

        return False
