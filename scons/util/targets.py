"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
from abc import ABC, abstractmethod
from typing import Callable

from SCons.Script.SConscript import SConsEnvironment as Environment


class Target(ABC):
    """
    An abstract base class for all targets of the build system.
    """

    def __init__(self, name: str):
        """
        :param name: The name of the target
        """
        self.name = name

    @abstractmethod
    def register(self, environment: Environment):
        """
        Must be implemented by subclasses in order to register the target.

        :param environment: The environment, the target should be registered at
        """


class PhonyTarget(Target):
    """
    A phony target, which executes a certain action and does not produce any output files.
    """

    def __init__(self, name: str, action: Callable[[], None]):
        """
        :param name:    The name of the target
        :param action:  The function to be executed by the target
        """
        super().__init__(name)
        self.action = action

    def register(self, environment: Environment):
        return environment.AlwaysBuild(environment.Alias(self.name, None, lambda **_: self.action()))
