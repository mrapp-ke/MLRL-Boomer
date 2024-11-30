"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
import sys

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, List, Set
from uuid import uuid4

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

        def __init__(self, parent_builder: Any):
            """
            :param parent_builder: The builder, this builder has been created from
            """
            self.parent_builder = parent_builder
            self.child_builders = []
            self.dependencies = set()

        def depends_on(self, *target_names: str) -> 'Target.Builder':
            """
            Adds on or several targets, this target should depend on.

            :param target_names:    The names of the targets, this target should depend on
            :return:                The `Target.Builder` itself
            """
            self.dependencies.update(target_names)
            return self

        def depends_on_build_target(self) -> 'BuildTarget.Builder':
            """
            Creates and returns a `BuildTarget.Builder` that allows to configure a build target, this target should
            depend on.

            :return: The `BuildTarget.Builder` that has been created
            """
            target_name = str(uuid4())
            target_builder = BuildTarget.Builder(self, target_name)
            self.dependencies.add(target_name)
            self.child_builders.append(target_builder)
            return target_builder

        def depends_on_phony_target(self) -> 'PhonyTarget.Builder':
            """
            Creates and returns a `PhonyTarget.Builder` that allows to configure a phony target, this target should
            depend on.

            :return: The `PhonyTarget.Builder` that has been created
            """
            target_name = str(uuid4())
            target_builder = PhonyTarget.Builder(self, target_name)
            self.child_builders.append(target_builder)
            self.dependencies.add(target_name)
            return target_builder

        @abstractmethod
        def _build(self, build_unit: BuildUnit) -> 'Target':
            """
            Must be implemented by subclasses in order to create the target that has been configured via the builder.

            :param build_unit:  The build unit, the target belongs to
            :return:            The target that has been created
            """

        def build(self, build_unit: BuildUnit) -> List['Target']:
            """
            Creates and returns all targets that have been configured via the builder.

            :param build_unit:  The build unit, the target belongs to
            :return:            The targets that have been created
            """
            return [self._build(build_unit)] + reduce(lambda aggr, builder: aggr + builder.build(build_unit),
                                                      self.child_builders, [])

    def __init__(self, name: str, dependencies: Set[str]):
        """
        :param name:            The name of the target
        :param dependencies:    The names of the targets, this target depends on
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


class BuildTarget(Target):
    """
    A build target, which executes a certain action and produces one or several output files.
    """

    class Runnable(ABC):
        """
        An abstract base class for all classes that can be run via a build target.
        """

        @abstractmethod
        def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
            """
            Must be implemented by subclasses in order to run the target.

            :param build_unit:  The build unit, the target belongs to
            :param modules:     A `ModuleRegistry` that can be used by the target for looking up modules
            """

        def get_output_files(self, modules: ModuleRegistry) -> List[str]:
            """
            May be overridden by subclasses in order to return the output files produced by the target.

            :param modules: A `ModuleRegistry` that can be used by the target for looking up modules
            :return:        A list that contains the output files
            """
            return []

    class Builder(Target.Builder):
        """
        A builder that allows to configure and create build targets.
        """

        def __init__(self, parent_builder: Any, name: str):
            """
            :param parent_builder:  The builder, this builder has been created from
            :param name:            The name of the target
            """
            super().__init__(parent_builder)
            self.name = name
            self.runnables = []

        def set_runnables(self, *runnables: 'BuildTarget.Runnable') -> Any:
            """
            Sets one or several `Runnable` objects to be run by the target.

            :param runnables:   The `Runnable` objects to be set
            :return:            The builder, this builder has been created from
            """
            self.runnables = list(runnables)
            return self.parent_builder

        def _build(self, build_unit: BuildUnit) -> Target:
            return BuildTarget(self.name, self.dependencies, self.runnables, build_unit)

    def __init__(self, name: str, dependencies: Set[str], runnables: List[Runnable], build_unit: BuildUnit):
        """
        :param name:            The name of the target or None, if the target does not have a name
        :param dependencies:    The names of the targets, this target depends on
        :param runnables:       The `BuildTarget.Runnable` to the be run by the target
        :param build_unit:      The `BuildUnit`, the target belongs to
        """
        super().__init__(name, dependencies)
        self.runnables = runnables
        self.build_unit = build_unit

    def register(self, environment: Environment, module_registry: ModuleRegistry) -> Any:

        def action():
            for runnable in self.runnables:
                runnable.run(self.build_unit, module_registry)

        output_files = reduce(lambda aggr, runnable: runnable.get_output_files(module_registry), self.runnables, [])
        target = (output_files if len(output_files) > 1 else output_files[0]) if output_files else None

        if target:
            return environment.Command(target, None, action=lambda **_: action())

        return environment.AlwaysBuild(environment.Alias(self.name, None, action=lambda **_: action()))


class PhonyTarget(Target):
    """
    A phony target, which executes a certain action and does not produce any output files.
    """

    Function = Callable[[], None]

    class Runnable(ABC):
        """
        An abstract base class for all classes that can be run via a phony target.
        """

        @abstractmethod
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

        def __init__(self, parent_builder: Any, name: str):
            """
            :param parent_builder:  The builder, this builder has been created from
            :param name:            The name of the target
            """
            super().__init__(parent_builder)
            self.name = name
            self.functions = []
            self.runnables = []

        def nop(self) -> Any:
            """
            Instructs the target to not execute any action.

            :return: The `TargetBuilder`, this builder has been created from
            """
            return self.parent_builder

        def set_functions(self, *functions: 'PhonyTarget.Function') -> Any:
            """
            Sets one or several functions to be run by the target.

            :param functions:   The functions to be set
            :return:            The builder, this builder has been created from
            """
            self.functions = list(functions)
            return self.parent_builder

        def set_runnables(self, *runnables: 'PhonyTarget.Runnable') -> Any:
            """
            Sets one or several `Runnable` objects to be run by the target.

            :param runnables:   The `Runnable` objects to be set
            :return:            The builder, this builder has been created from
            """
            self.runnables = list(runnables)
            return self.parent_builder

        def _build(self, build_unit: BuildUnit) -> Target:

            def action(module_registry: ModuleRegistry):
                for function in self.functions:
                    function()

                for runnable in self.runnables:
                    runnable.run(build_unit, module_registry)

            return PhonyTarget(self.name, self.dependencies, action)

    def __init__(self, name: str, dependencies: Set[str], action: Callable[[ModuleRegistry], None]):
        """
        :param name:            The name of the target
        :param dependencies:    The names of the targets, this target depends on
        :param action:          The action to be executed by the target
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

    def add_build_target(self, name: str) -> BuildTarget.Builder:
        """
        Adds a build target.

        :param name:    The name of the target
        :return:        A `BuildTarget.Builder` that allows to configure the target
        """
        target_builder = BuildTarget.Builder(self, name)
        self.target_builders.append(target_builder)
        return target_builder

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
        return reduce(lambda aggr, builder: aggr + builder.build(self.build_unit), self.target_builders, [])


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
