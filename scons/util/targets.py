"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from uuid import uuid4

from util.modules import ModuleRegistry
from util.units import BuildUnit

from SCons.Script import COMMAND_LINE_TARGETS
from SCons.Script.SConscript import SConsEnvironment as Environment


class Target(ABC):
    """
    An abstract base class for all targets of the build system.
    """

    @dataclass
    class Dependency:
        """
        A single dependency of a parent target.

        Attributes:
            target_name:        The name of the target, the parent target depends on
            clean_dependency:   True, if the output files of the dependency should also be cleaned when cleaning the
                                output files of the parent target, False otherwise
        """
        target_name: str
        clean_dependency: bool

        def __eq__(self, other: 'Target.Dependency') -> bool:
            return self.target_name == other.target_name

        def __hash__(self) -> int:
            return hash(self.target_name)

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

        def depends_on(self, *target_names: str, clean_dependencies: bool = False) -> 'Target.Builder':
            """
            Adds on or several targets, this target should depend on.

            :param target_names:        The names of the targets, this target should depend on
            :param clean_dependencies:  True, if output files of the dependencies should also be cleaned when cleaning
                                        the output files of this target, False otherwise
            :return:                    The `Target.Builder` itself
            """
            for target_name in target_names:
                self.dependencies.add(Target.Dependency(target_name=target_name, clean_dependency=clean_dependencies))

            return self

        def depends_on_build_target(self, clean_dependency: bool = True) -> 'BuildTarget.Builder':
            """
            Creates and returns a `BuildTarget.Builder` that allows to configure a build target, this target should
            depend on.

            :param clean_dependency:    True, if output files of the dependency should also be cleaned when cleaning the
                                        output files of this target, False otherwise
            :return:                    The `BuildTarget.Builder` that has been created
            """
            target_name = str(uuid4())
            target_builder = BuildTarget.Builder(self, target_name)
            self.child_builders.append(target_builder)
            self.depends_on(target_name, clean_dependencies=clean_dependency)
            return target_builder

        def depends_on_build_targets(self,
                                     iterable: Iterable[Any],
                                     target_configurator: Callable[[Any, 'BuildTarget.Builder'], None],
                                     clean_dependencies: bool = True) -> 'Target.Builder':
            """
            Configures multiple build targets, one for each value in a given `Iterable`, this target should depend on.

            :param iterable:            An `Iterable` that provides access to the values for which dependencies should
                                        be created
            :param target_configurator: A function that accepts one value in the `Iterable` at a time, as well as a
                                        `BuildTarget.Builder` for configuring the corresponding dependency
            :param clean_dependencies:  True, if output files of the dependencies should also be cleaned when cleaning
                                        the output files of this target, False otherwise
            :return:                    The `Target.Builder` itself
            """
            for value in iterable:
                target_builder = self.depends_on_build_target(clean_dependency=clean_dependencies)
                target_configurator(value, target_builder)

            return self

        def depends_on_phony_target(self, clean_dependency: bool = True) -> 'PhonyTarget.Builder':
            """
            Creates and returns a `PhonyTarget.Builder` that allows to configure a phony target, this target should
            depend on.

            :param clean_dependency:    True, if output files of the dependency should also be cleaned when cleaning the
                                        output files of this target, False otherwise
            :return:                    The `PhonyTarget.Builder` that has been created
            """
            target_name = str(uuid4())
            target_builder = PhonyTarget.Builder(self, target_name)
            self.child_builders.append(target_builder)
            self.depends_on(target_name, clean_dependencies=clean_dependency)
            return target_builder

        def depends_on_phony_targets(self,
                                     iterable: Iterable[Any],
                                     target_configurator: Callable[[Any, 'PhonyTarget.Builder'], None],
                                     clean_dependencies: bool = True) -> 'Target.Builder':
            """
            Configures multiple phony targets, one for each value in a given `Iterable`, this target should depend on.

            :param iterable:            An `Iterable` that provides access to the values for which dependencies should
                                        be created
            :param target_configurator: A function that accepts one value in the `Iterable` at a time, as well as a
                                        `BuildTarget.Builder` for configuring the corresponding dependency
            :param clean_dependencies:  True, if output files of the dependencies should also be cleaned when cleaning
                                        the output files of this target, False otherwise
            :return:                    The `Target.Builder` itself
            """
            for value in iterable:
                target_builder = self.depends_on_phony_target(clean_dependency=clean_dependencies)
                target_configurator(value, target_builder)

            return self

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

    def __init__(self, name: str, dependencies: Set['Target.Dependency']):
        """
        :param name:            The name of the target
        :param dependencies:    The dependencies of the target
        """
        self.name = name
        self.dependencies = dependencies

    @abstractmethod
    def register(self, environment: Environment, module_registry: ModuleRegistry) -> Any:
        """
        Must be implemented by subclasses in order to register this target.

        :param environment:     The environment, the target should be registered at
        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        :return:                The scons target that has been created
        """

    def get_clean_files(self, module_registry: ModuleRegistry) -> Optional[List[str]]:
        """
        May be overridden by subclasses in order to return the files that should be cleaned up for this target.

        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        :return:                A list that contains the files to be cleaned or None, if cleaning is not necessary
        """
        return None


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

        def get_input_files(self, modules: ModuleRegistry) -> List[str]:
            """
            May be overridden by subclasses in order to return the input files required by the target.

            :param modules: A `ModuleRegistry` that can be used by the target for looking up modules
            :return:        A list that contains the input files
            """
            return []

        def get_output_files(self, modules: ModuleRegistry) -> List[str]:
            """
            May be overridden by subclasses in order to return the output files produced by the target.

            :param modules: A `ModuleRegistry` that can be used by the target for looking up modules
            :return:        A list that contains the output files
            """
            return []

        def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
            """
            May be overridden by subclasses in order to return the output files produced by the target that must be
            cleaned.

            :param modules: A `ModuleRegistry` that can be used by the target for looking up modules
            :return:        A list that contains the files to be cleaned
            """
            return self.get_output_files(modules)

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

    def __init__(self, name: str, dependencies: Set[Target.Dependency], runnables: List[Runnable],
                 build_unit: BuildUnit):
        """
        :param name:            The name of the target or None, if the target does not have a name
        :param dependencies:    The dependencies of the target
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

        input_files = reduce(lambda aggr, runnable: runnable.get_input_files(module_registry), self.runnables, [])
        source = (input_files if len(input_files) > 1 else input_files[0]) if input_files else None
        output_files = reduce(lambda aggr, runnable: runnable.get_output_files(module_registry), self.runnables, [])
        target = (output_files if len(output_files) > 1 else output_files[0]) if output_files else None

        if target:
            return environment.Depends(
                environment.Alias(self.name, None, None),
                environment.Command(target, source, action=lambda **_: action()),
            )

        return environment.AlwaysBuild(environment.Alias(self.name, None, action=lambda **_: action()))

    def get_clean_files(self, module_registry: ModuleRegistry) -> Optional[List[str]]:
        return reduce(lambda aggr, runnable: runnable.get_clean_files(module_registry), self.runnables, [])


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

    def __init__(self, name: str, dependencies: Set[Target.Dependency], action: Callable[[ModuleRegistry], None]):
        """
        :param name:            The name of the target
        :param dependencies:    The dependencies of the target
        :param action:          The action to be executed by the target
        """
        super().__init__(name, dependencies)
        self.action = action

    def register(self, environment: Environment, module_registry: ModuleRegistry) -> Any:
        return environment.AlwaysBuild(
            environment.Alias(self.name, None, action=lambda **_: self.action(module_registry)),
        )


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

    def __register_scons_targets(self) -> Dict[str, Any]:
        scons_targets_by_name = {}

        for target_name, target in self.targets_by_name.items():
            scons_targets_by_name[target_name] = target.register(self.environment, self.module_registry)

        return scons_targets_by_name

    def __register_scons_dependencies(self, scons_targets_by_name: Dict[str, Any]):
        for target_name, target in self.targets_by_name.items():
            scons_target = scons_targets_by_name[target_name]
            scons_dependencies = []

            for dependency in target.dependencies:
                try:
                    scons_dependencies.append(scons_targets_by_name[dependency.target_name])
                except KeyError:
                    print('Dependency "' + dependency.target_name + '" of target "' + target_name
                          + '" has not been registered')
                    sys.exit(-1)

            if scons_dependencies:
                self.environment.Depends(scons_target, scons_dependencies)

    def __get_parent_targets(self, *target_names: str) -> Set[str]:
        result = set()
        parent_targets = reduce(lambda aggr, target_name: aggr | self.parent_targets_by_name.get(target_name, set()),
                                target_names, set())

        if parent_targets:
            result.update(self.__get_parent_targets(*parent_targets))

        result.update(parent_targets)
        return result

    def __register_scons_clean_targets(self, scons_targets_by_name: Dict[str, Any]):
        if self.environment.GetOption('clean'):
            for target_name, target in self.targets_by_name.items():
                parent_targets = {target_name} | self.__get_parent_targets(target_name)

                if not COMMAND_LINE_TARGETS or reduce(
                        lambda aggr, parent_target: aggr or parent_target in COMMAND_LINE_TARGETS, parent_targets,
                        False):
                    clean_files = target.get_clean_files(self.module_registry)

                    if clean_files:
                        clean_targets = [scons_targets_by_name[parent_target] for parent_target in parent_targets]
                        self.environment.Clean(clean_targets, clean_files)

    def __init__(self, module_registry: ModuleRegistry):
        """
        :param module_registry: The `ModuleRegistry` that should be used by targets for looking up modules
        """
        self.environment = Environment()
        self.module_registry = module_registry
        self.targets_by_name = {}
        self.parent_targets_by_name = {}

    def add_target(self, target: Target):
        """
        Adds a new target to be registered.

        :param target: The target to be added
        """
        self.targets_by_name[target.name] = target

        for dependency in target.dependencies:
            if dependency.clean_dependency:
                parent_targets = self.parent_targets_by_name.setdefault(dependency.target_name, set())
                parent_targets.add(target.name)

    def register(self):
        """
        Registers all targets that have previously been added.
        """
        scons_targets_by_name = self.__register_scons_targets()
        self.__register_scons_dependencies(scons_targets_by_name)
        self.__register_scons_clean_targets(scons_targets_by_name)

    @property
    def target_names(self) -> Set[str]:
        """
        A set that contains the names of all targets that have previously been added.
        """
        return set(self.targets_by_name.keys())
