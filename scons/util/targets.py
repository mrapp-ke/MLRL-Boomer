"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
import sys

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Set

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

        def __init__(self):
            self.dependencies = set()

        def depends_on(self, *target_names: str) -> 'Target.Builder':
            """
            Adds on or several targets, this target should depend on.

            :param target_names:    The names of the targets, this target should depend on
            :return:                The `Target.Builder` itself
            """
            self.dependencies.update(target_names)
            return self

        @abstractmethod
        def build(self) -> 'Target':
            """
            Creates and returns the target that has been configured via the builder.

            :return: The target that has been created
            """

    def __init__(self, name: str, dependencies: Set[str]):
        """
        :param name:            The name of the target
        :param dependencies:    The name of the targets, this target depends on
        """
        self.name = name
        self.dependencies = dependencies

    @abstractmethod
    def register(self, environment: Environment, module_registry: ModuleRegistry) -> Any:
        """
        Must be implemented by subclasses in order to register the target.

        :param environment:     The environment, the target should be registered at
        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        :return:                The scons target that has been created
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
            super().__init__()
            self.target_builder = target_builder
            self.name = name
            self.functions = []
            self.runnables = []

        def nop(self) -> 'TargetBuilder':
            """
            Instructs the target to not execute any action.

            :return: The `TargetBuilder`, this builder has been created from
            """
            return self.target_builder

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

            return PhonyTarget(self.name, self.dependencies, action)

    def __init__(self, name: str, dependencies: Set[str], action: Callable[[ModuleRegistry], None]):
        """
        :param name:    The name of the target
        :param action:  The action to be executed by the target
        """
        super().__init__(name, dependencies)
        self.action = action

    def register(self, environment: Environment, module_registry: ModuleRegistry) -> Any:
        return environment.AlwaysBuild(
            environment.Alias(self.name, None, action=lambda **_: self.action(module_registry)))


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


class TargetRegistry:
    """
    Allows to register targets.
    """

    def __init__(self, module_registry: ModuleRegistry):
        """
        :param module_registry: The `ModuleRegistry` that should be used by targets for looking up modules
        """
        self.environment = Environment()
        self.module_registry = module_registry
        self.targets_by_name = {}

    def add_target(self, target: Target):
        """
        Adds a new target to be registered.

        :param target: The target to be added
        """
        self.targets_by_name[target.name] = target

    def register(self):
        """
        Registers all targets that have previously been added.
        """
        scons_targets_by_name = {}

        for target_name, target in self.targets_by_name.items():
            scons_targets_by_name[target_name] = target.register(self.environment, self.module_registry)

        for target_name, target in self.targets_by_name.items():
            scons_target = scons_targets_by_name[target_name]
            scons_dependencies = []

            for dependency in target.dependencies:
                try:
                    scons_dependencies.append(scons_targets_by_name[dependency])
                except KeyError:
                    print('Dependency "' + dependency + '" of target "' + target_name + '" has not been registered')
                    sys.exit(-1)

            if scons_dependencies:
                self.environment.Depends(scons_target, scons_dependencies)
