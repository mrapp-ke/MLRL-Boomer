"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
from abc import ABC, abstractmethod
from typing import Callable

from util.modules import ModuleRegistry

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
    def register(self, environment: Environment, module_registry: ModuleRegistry):
        """
        Must be implemented by subclasses in order to register the target.

        :param environment:     The environment, the target should be registered at
        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        """


class PhonyTarget(Target):
    """
    A phony target, which executes a certain action and does not produce any output files.
    """

    Function = Callable[[], None]

    class Runnable(ABC):
        """
        An abstract base class for all classes that can be run via a phony target.
        """

        def run(self, modules: ModuleRegistry):
            """
            Must be implemented by subclasses in order to run the target.

            :param modules: A `ModuleRegistry` that can be used by the target for looking up modules
            """

    class Builder:
        """
        A builder that allows to configure and create phony targets.
        """

        def __init__(self, name: str):
            """
            :param name: The name of the target
            """
            self.name = name
            self.function = None
            self.runnable = None

        def set_function(self, function: 'PhonyTarget.Function') -> 'PhonyTarget.Builder':
            """
            Sets a function to be run by the target.

            :param function:    The function to be run
            :return:            The `PhonyTarget.Builder` itself
            """
            self.function = function
            return self

        def set_runnable(self, runnable: 'PhonyTarget.Runnable') -> 'PhonyTarget.Builder':
            """
            Sets a runnable to be run by the target.

            :param runnable:    The runnable to be run
            :return:            The `PhonyTarget.Builder` itself
            """
            self.runnable = runnable
            return self

        def build(self) -> 'PhonyTarget':
            """
            Creates and returns the phony target that has been configured via the builder.

            :return: The phony target that has been created
            """

            def action(module_registry: ModuleRegistry):
                if self.function:
                    self.function()

                if self.runnable:
                    self.runnable.run(module_registry)

            return PhonyTarget(self.name, action)

    def __init__(self, name: str, action: Callable[[ModuleRegistry], None]):
        """
        :param name:    The name of the target
        :param action:  The action to be executed by the target
        """
        super().__init__(name)
        self.action = action

    def register(self, environment: Environment, module_registry: ModuleRegistry):
        environment.AlwaysBuild(environment.Alias(self.name, None, action=lambda **_: self.action(module_registry)))
