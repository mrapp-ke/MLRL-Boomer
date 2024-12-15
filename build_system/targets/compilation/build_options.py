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

    def __init__(self, name: str, subpackage: Optional[str]):
        """
        name:       The name of the build option
        subpackage: The subpackage, the build option corresponds to, or None, if it is a global option
        """
        self.name = name
        self.subpackage = subpackage

    @property
    def key(self) -> str:
        """
        The key to be used for setting the build option.
        """
        return (self.subpackage + ':' if self.subpackage else '') + self.name

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


class EnvBuildOption(BuildOption):
    """
    A build option, whose value is obtained from an environment variable.
    """

    def __init__(self, name: str, subpackage: Optional[str] = None):
        super().__init__(name, subpackage)

    @property
    def value(self) -> Optional[str]:
        value = get_env(environ, self.name.upper(), None)

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
