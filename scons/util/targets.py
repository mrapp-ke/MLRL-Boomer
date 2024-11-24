"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
from abc import ABC, abstractmethod
from typing import Callable, List

from util.modules import ModuleRegistry
from util.units import BuildUnit

from SCons.Script.SConscript import SConsEnvironment as Environment


class Target(ABC):
    """
    An abstract base class for all targets of the build system.
    """

    class Builder(ABC):
        """
        An abstract base class for all builders that allow to configure and create targets.
        """

        @abstractmethod
        def build(self) -> 'Target':
            """
            Creates and returns the target that has been configured via the builder.

            :return: The target that has been created
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

        def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
            """
            Must be implemented by subclasses in order to run the target.

            :param build_unit:  The build unit, the target belongs to
            :param modules:     A `ModuleRegistry` that can be used by the target for looking up modules
            """

    class Builder(Target.Builder):
        """
        A builder that allows to configure and create phony targets.
        """

        def __init__(self, target_builder: 'TargetBuilder', name: str):
            """
            :param target_builder:  The `TargetBuilder`, this builder has been created from
            :param name:            The name of the target
            """
            self.target_builder = target_builder
            self.name = name
            self.functions = []
            self.runnables = []

        def set_functions(self, *functions: 'PhonyTarget.Function') -> 'TargetBuilder':
            """
            Sets one or several functions to be run by the target.

            :param functions:   The functions to be set
            :return:            The `TargetBuilder`, this builder has been created from
            """
            self.functions = list(functions)
            return self.target_builder

        def set_runnables(self, *runnables: 'PhonyTarget.Runnable') -> 'TargetBuilder':
            """
            Sets one or several `Runnable` objects to be run by the target.

            :param runnables:   The `Runnable` objects to be set
            :return:            The `TargetBuilder`, this builder has been created from
            """
            self.runnables = list(runnables)
            return self.target_builder

        def build(self) -> Target:

            def action(module_registry: ModuleRegistry):
                for function in self.functions:
                    function()

                for runnable in self.runnables:
                    runnable.run(self.target_builder.build_unit, module_registry)

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


class TargetBuilder:
    """
    A builder that allows to configure and create multiple targets.
    """

    def __init__(self, build_unit: BuildUnit = BuildUnit()):
        """
        :param build_unit: The build unit, the targets belong to
        """
        self.build_unit = build_unit
        self.target_builders = []

    def add_phony_target(self, name: str) -> PhonyTarget.Builder:
        """
        Adds a phony target.

        :param name:    The name of the target
        :return:        A `PhonyTarget.Builder` that allows to configure the target
        """
        target_builder = PhonyTarget.Builder(self, name)
        self.target_builders.append(target_builder)
        return target_builder

    def build(self) -> List[Target]:
        """
        Creates and returns the targets that have been configured via the builder.

        :return: A list that stores the targets that have been created
        """
        return [target_builder.build() for target_builder in self.target_builders]
