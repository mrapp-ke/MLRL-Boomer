"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for defining individual targets of the build process.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from core.build_unit import BuildUnit
from core.changes import ChangeDetection
from core.modules import Module, ModuleRegistry
from util.format import format_iterable
from util.io import delete_files
from util.log import Log


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
        clean_dependency: bool = True

        def __str__(self) -> str:
            return self.target_name

        def __eq__(self, other) -> bool:
            return type(self) == type(other) and self.target_name == other.target_name

    class Builder(ABC):
        """
        An abstract base class for all builders that allow to configure and create targets.
        """

        def __anonymous_target_name(self) -> str:
            return 'anonymous-dependency-' + str(len(self.dependencies) + 1) + '-of-' + self.target_name

        def __init__(self, parent_builder: Any, target_name: str):
            """
            :param parent_builder:  The builder, this builder has been created from
            :param target_name:     The name of the target that is configured via the builder
            """
            self.parent_builder = parent_builder
            self.target_name = target_name
            self.child_builders = []
            self.dependency_names = set()
            self.dependencies = []

        def depends_on(self, *target_names: str, clean_dependencies: bool = False) -> 'Target.Builder':
            """
            Adds on or several targets, this target should depend on.

            :param target_names:        The names of the targets, this target should depend on
            :param clean_dependencies:  True, if output files of the dependencies should also be cleaned when cleaning
                                        the output files of this target, False otherwise
            :return:                    The `Target.Builder` itself
            """
            for target_name in target_names:
                if not target_name in self.dependency_names:
                    self.dependency_names.add(target_name)
                    self.dependencies.append(
                        Target.Dependency(target_name=target_name, clean_dependency=clean_dependencies))

            return self

        def depends_on_build_target(self, clean_dependency: bool = True) -> 'BuildTarget.Builder':
            """
            Creates and returns a `BuildTarget.Builder` that allows to configure a build target, this target should
            depend on.

            :param clean_dependency:    True, if output files of the dependency should also be cleaned when cleaning the
                                        output files of this target, False otherwise
            :return:                    The `BuildTarget.Builder` that has been created
            """
            target_name = self.__anonymous_target_name()
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
            target_name = self.__anonymous_target_name()
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

    def __init__(self, name: str, dependencies: List['Target.Dependency']):
        """
        :param name:            The name of the target
        :param dependencies:    A list that contains all dependencies of the target
        """
        self.name = name
        self.dependencies = dependencies

    @abstractmethod
    def run(self, module_registry: ModuleRegistry):
        """
        Must be implemented by subclasses in order to run this target.

        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        """

    def clean(self, module_registry: ModuleRegistry):
        """
        May be overridden by subclasses in order to clean up this target.

        :param module_registry: The `ModuleRegistry` that can be used by the target for looking up modules
        """

    def __str__(self) -> str:
        result = type(self).__name__ + '{name="' + self.name + '"'

        if self.dependencies:
            result += ', dependencies={' + format_iterable(self.dependencies, delimiter='"') + '}'

        return result + '}'


class BuildTarget(Target):
    """
    A build target, which produces one or several output files from given input files.
    """

    class Runnable(ABC):
        """
        An abstract base class for all classes that can be run via a build target.
        """

        def __init__(self, module_filter: Module.Filter):
            """
            :param module_filter: A filter that matches the modules, the target should be applied to
            """
            self.module_filter = module_filter

        @abstractmethod
        def run(self, build_unit: BuildUnit, module: Module):
            """
            may be overridden by subclasses in order to apply the target to an individual module that matches the
            filter.

            :param build_unit:  The build unit, the target belongs to
            :param module:      The module, the target should be applied to
            """

        def get_input_files(self, module: Module) -> List[str]:
            """
            May be overridden by subclasses in order to return the input files required by the target.

            :param module:  The module, the target should be applied to
            :return:        A list that contains the input files
            """
            return []

        def get_output_files(self, module: Module) -> List[str]:
            """
            May be overridden by subclasses in order to return the output files produced by the target.

            :param module:  The module, the target should be applied to
            :return:        A list that contains the output files
            """
            return []

        def get_clean_files(self, module: Module) -> List[str]:
            """
            May be overridden by subclasses in order to return the output files produced by the target that must be
            cleaned.

            :param module:  The module, the target should be applied to
            :return:        A list that contains the files to be cleaned
            """
            return self.get_output_files(module)

    class Builder(Target.Builder):
        """
        A builder that allows to configure and create build targets.
        """

        def __init__(self, parent_builder: Any, target_name: str):
            """
            :param parent_builder:  The builder, this builder has been created from
            :param target_name:     The name of the target that is configured via the builder
            """
            super().__init__(parent_builder, target_name)
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
            return BuildTarget(self.target_name, self.dependencies, self.runnables, build_unit)

    def __get_missing_output_files(self, runnable: Runnable, module: Module) -> Tuple[List[str], List[str]]:
        output_files = runnable.get_output_files(module)
        missing_output_files = [output_file for output_file in output_files if not path.exists(output_file)]

        if output_files:
            if missing_output_files:
                Log.verbose(
                    'Target "%s" must be applied to module "%s", because the following output files do not exist:\n',
                    self.name, str(module))

                for missing_output_file in missing_output_files:
                    Log.verbose(' - %s', missing_output_file)

                Log.verbose('')
            else:
                Log.verbose('Target "%s" must not be applied to module "%s", because all output files already exist:\n',
                            self.name, str(module))

                for output_file in output_files:
                    Log.verbose(' - %s', output_file)

                Log.verbose('')

        return output_files, missing_output_files

    def __get_changed_input_files(self, runnable: Runnable, module: Module) -> Tuple[List[str], List[str]]:
        input_files = runnable.get_input_files(module)
        changed_input_files = [
            input_file for input_file in input_files if self.change_detection.get_changed_files(module, *input_files)
        ]

        if input_files:
            if changed_input_files:
                Log.verbose(
                    'Target "%s" must be applied to module "%s", because the following input files have changed:\n',
                    self.name, str(module))

                for changed_input_file in changed_input_files:
                    Log.verbose(' - %s', changed_input_file)

                Log.verbose('')
            else:
                Log.verbose('Target "%s" must not be applied to module "%s", because no input files have changed:\n',
                            self.name, str(module))

                for input_file in input_files:
                    Log.verbose(' - %s', input_file)

                Log.verbose('')

        return input_files, changed_input_files

    def __init__(self, name: str, dependencies: List[Target.Dependency], runnables: List[Runnable],
                 build_unit: BuildUnit):
        """
        :param name:            The name of the target or None, if the target does not have a name
        :param dependencies:    A list that contains all dependencies of the target
        :param runnables:       The `BuildTarget.Runnable` to the be run by the target
        :param build_unit:      The `BuildUnit`, the target belongs to
        """
        super().__init__(name, dependencies)
        self.runnables = runnables
        self.build_unit = build_unit
        self.change_detection = ChangeDetection(path.join('build_system', 'build', self.name + '.json'))

    def run(self, module_registry: ModuleRegistry):
        for runnable in self.runnables:
            modules = module_registry.lookup(runnable.module_filter)

            for module in modules:
                output_files, missing_output_files = self.__get_missing_output_files(runnable, module)
                input_files, changed_input_files = self.__get_changed_input_files(runnable, module)

                if (not output_files and not input_files) or missing_output_files or changed_input_files:
                    runnable.run(self.build_unit, module)
                    self.change_detection.track_files(module, *input_files)

    def clean(self, module_registry: ModuleRegistry):
        for runnable in self.runnables:
            modules = module_registry.lookup(runnable.module_filter)

            for module in modules:
                clean_files = runnable.get_clean_files(module)
                delete_files(*clean_files, accept_missing=True)


class PhonyTarget(Target):
    """
    A phony target, which executes a certain action unconditionally.
    """

    Function = Callable[[], None]

    class Runnable(ABC):
        """
        An abstract base class for all classes that can be run via a phony target.
        """

        def __init__(self, module_filter: Module.Filter):
            """
            :param module_filter: A filter that matches the modules, the target should be applied to
            """
            self.module_filter = module_filter

        def run_all(self, build_unit: BuildUnit, module: List[Module]):
            """
            May be overridden by subclasses in order to apply the target to all modules that match the filter.

            :param build_unit:  The build unit, the target belongs to
            :param module:      A list that contains the modules, the target should be applied to
            """
            raise NotImplementedError('Class ' + type(self).__name__ + ' does not implement the "run_all" method')

        def run(self, build_unit: BuildUnit, module: Module):
            """
            May be overridden by subclasses in order to apply the target to an individual module that matches the
            filter.

            :param build_unit:  The build unit, the target belongs to
            :param module:      The module, the target should be applied to
            """
            raise NotImplementedError('Class ' + type(self).__name__ + ' does not implement the "run" method')

    class Builder(Target.Builder):
        """
        A builder that allows to configure and create phony targets.
        """

        def __init__(self, parent_builder: Any, target_name: str):
            """
            :param parent_builder:  The builder, this builder has been created from
            :param target_name:     The name of the target that is configured via the builder
            """
            super().__init__(parent_builder, target_name)
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
                    modules = module_registry.lookup(runnable.module_filter)

                    try:
                        runnable.run_all(build_unit, modules)
                    except NotImplementedError:
                        try:
                            for module in modules:
                                runnable.run(build_unit, module)
                        except NotImplementedError as error:
                            raise RuntimeError('Class ' + type(runnable).__name__
                                               + ' must implement either the "run_all" or "run" method') from error

            return PhonyTarget(self.target_name, self.dependencies, action)

    def __init__(self, name: str, dependencies: List[Target.Dependency], action: Callable[[ModuleRegistry], None]):
        """
        :param name:            The name of the target
        :param dependencies:    A list that contains all dependencies of the target
        :param action:          The action to be executed by the target
        """
        super().__init__(name, dependencies)
        self.action = action

    def run(self, module_registry: ModuleRegistry):
        self.action(module_registry)


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


class DependencyGraph:
    """
    A graph that determines the execution order of targets based on the dependencies between them.
    """

    @dataclass
    class Node(ABC):
        """
        An abstract base class for all nodes in a dependency graph.

        Attributes:
            target: The target that corresponds to this node
            parent: The parent of this node, if any
            child:  The child of this node, if any
        """
        target: Target
        parent: Optional['DependencyGraph.Node'] = None
        child: Optional['DependencyGraph.Node'] = None

        @staticmethod
        def from_name(targets_by_name: Dict[str, Target], target_name: str, clean: bool) -> 'DependencyGraph.Node':
            """
            Creates and returns a new node of a dependency graph corresponding to the target with a specific name.

            :param targets_by_name: A dictionary that stores all available targets by their names
            :param target_name:     The name of the target, the node should correspond to
            :param clean:           True, if the target should be cleaned, False otherwise
            :return:                The node that has been created
            """
            target = targets_by_name[target_name]
            return DependencyGraph.CleanNode(target) if clean else DependencyGraph.RunNode(target)

        @staticmethod
        def from_dependency(targets_by_name: Dict[str, Target], dependency: Target.Dependency,
                            clean: bool) -> Optional['DependencyGraph.Node']:
            """
            Creates and returns a new node of a dependency graph corresponding to the target referred to by a
            `Target.Dependency`.

            :param targets_by_name: A dictionary that stores all available targets by their names
            :param dependency:      The dependency referring to the target, the node should correspond to
            :param clean:           True, if the target should be cleaned, False otherwise
            :return:                The node that has been created or None, if the dependency does not require a node to
                                    be created
            """
            if not clean or dependency.clean_dependency:
                target = targets_by_name[dependency.target_name]
                return DependencyGraph.CleanNode(target) if clean else DependencyGraph.RunNode(target)

            return None

        @abstractmethod
        def execute(self, module_registry: ModuleRegistry):
            """
            Must be implemented by subclasses in order to execute the node.
            """

        @abstractmethod
        def copy(self) -> 'DependencyGraph.Node':
            """
            Must be implemented by subclasses in order to create a shallow copy of the node.

            :return: The copy that has been created
            """

        def __str__(self) -> str:
            return '[' + self.target.name + ']'

        def __eq__(self, other) -> bool:
            return type(self) == type(other) and self.target == other.target

    class RunNode(Node):
        """
        A node in the dependency graph that runs one or several targets.
        """

        def execute(self, module_registry: ModuleRegistry):
            Log.verbose('Running target "%s"...', self.target.name)
            self.target.run(module_registry)

        def copy(self) -> 'DependencyGraph.Node':
            return DependencyGraph.RunNode(self.target)

    class CleanNode(Node):
        """
        A node in the dependency graph that cleans one or several targets.
        """

        def execute(self, module_registry: ModuleRegistry):
            Log.verbose('Cleaning target "%s"...', self.target.name)
            self.target.clean(module_registry)

        def copy(self) -> 'DependencyGraph.Node':
            return DependencyGraph.CleanNode(self.target)

    @dataclass
    class Sequence:
        """
        A sequence consisting of several nodes in a dependency graph.

        Attributes:
            first:  The first node in the sequence
            last:   The last node in the sequence
        """
        first: 'DependencyGraph.Node'
        last: 'DependencyGraph.Node'

        @staticmethod
        def from_node(node: 'DependencyGraph.Node') -> 'DependencyGraph.Sequence':
            """
            Creates and returns a new path that consists of a single node.

            :param node:    The node
            :return:        The path that has been created
            """
            node.parent = None
            node.child = None
            return DependencyGraph.Sequence(first=node, last=node)

        def prepend(self, node: 'DependencyGraph.Node'):
            """
            Adds a new node at the start of the sequence.

            :param node: The node to be added
            """
            first_node = self.first
            first_node.parent = node
            node.parent = None
            node.child = first_node
            self.first = node

        def copy(self) -> 'DependencyGraph.Sequence':
            """
            Creates a deep copy of the sequence.

            :return: The copy that has been created
            """
            current_node = self.last
            copy = DependencyGraph.Sequence.from_node(current_node.copy())
            current_node = current_node.parent

            while current_node:
                copy.prepend(current_node.copy())
                current_node = current_node.parent

            return copy

        def execute(self, module_registry: ModuleRegistry):
            current_node = self.first

            while current_node:
                current_node.execute(module_registry)
                current_node = current_node.child

        def __str__(self) -> str:
            current_node = self.first
            result = ' → ' + str(current_node)
            current_node = current_node.child
            indent = 1

            while current_node:
                result += '\n' + reduce(lambda aggr, _: aggr + '   ', range(indent), '') + ' ↳ ' + str(current_node)
                current_node = current_node.child
                indent += 1

            return result

    @staticmethod
    def __expand_sequence(targets_by_name: Dict[str, Target], sequence: Sequence, clean: bool) -> List[Sequence]:
        sequences = []
        dependencies = sequence.first.target.dependencies

        if dependencies:
            for dependency in dependencies:
                new_node = DependencyGraph.Node.from_dependency(targets_by_name, dependency, clean=clean)

                if new_node:
                    new_sequence = sequence.copy()
                    new_sequence.prepend(new_node)
                    sequences.extend(DependencyGraph.__expand_sequence(targets_by_name, new_sequence, clean=clean))
                else:
                    sequences.append(sequence)
        else:
            sequences.append(sequence)

        return sequences

    @staticmethod
    def __create_sequence(targets_by_name: Dict[str, Target], target_name: str, clean: bool) -> List[Sequence]:
        node = DependencyGraph.Node.from_name(targets_by_name, target_name, clean=clean)
        sequence = DependencyGraph.Sequence.from_node(node)
        return DependencyGraph.__expand_sequence(targets_by_name, sequence, clean=clean)

    @staticmethod
    def __find_in_parents(node: Node, parent: Optional[Node]) -> Optional[Node]:
        while parent:
            if parent == node:
                return parent

            parent = parent.parent

        return None

    @staticmethod
    def __merge_two_sequences(first_sequence: Sequence, second_sequence: Sequence) -> Sequence:
        first_node = first_sequence.last
        second_node = second_sequence.last

        while second_node:
            overlapping_node = DependencyGraph.__find_in_parents(second_node, first_node)

            if overlapping_node:
                first_node = overlapping_node.parent
            else:
                new_node = second_node.copy()

                if first_node:
                    new_node.parent = first_node

                    if first_node.child:
                        new_node.child = first_node.child
                        first_node.child.parent = new_node

                    first_node.child = new_node

                    if first_node == first_sequence.last:
                        first_sequence.last = new_node
                else:
                    first_sequence.prepend(new_node)

            second_node = second_node.parent

        return first_sequence

    @staticmethod
    def __merge_multiple_sequences(sequences: List[Sequence]) -> Sequence:
        while len(sequences) > 1:
            second_sequence = sequences.pop()
            first_sequence = sequences.pop()
            merged_sequence = DependencyGraph.__merge_two_sequences(first_sequence, second_sequence)
            sequences.append(merged_sequence)

        return sequences[0]

    def __init__(self, targets_by_name: Dict[str, Target], *target_names: str, clean: bool):
        """
        :param targets_by_name: A dictionary that stores all available targets by their names
        :param target_names:    The names of the targets to be included in the graph
        :param clean:           True, if the targets should be cleaned, False otherwise
        """
        self.sequence = self.__merge_multiple_sequences(
            reduce(lambda aggr, target_name: aggr + self.__create_sequence(targets_by_name, target_name, clean=clean),
                   target_names, []))

    def execute(self, module_registry: ModuleRegistry):
        """
        Executes all targets in the graph in the pre-determined order.

        :param module_registry: The `ModuleRegistry` that should be used by targets for looking up modules
        """
        self.sequence.execute(module_registry)

    def __str__(self) -> str:
        return str(self.sequence)


class TargetRegistry:
    """
    Allows to register targets.
    """

    def __init__(self):
        self.targets_by_name = {}

    def register(self, target: Target):
        """
        Registers a new target.

        :param target: The target to be registered
        """
        existing = self.targets_by_name.get(target.name)

        if existing:
            raise ValueError('Failed to register target ' + str(target)
                             + ', because a target with the same name has already been registered: ' + str(existing))

        self.targets_by_name[target.name] = target

    def create_dependency_graph(self, *target_names: str, clean: bool = False) -> DependencyGraph:
        """
        Creates and returns a `DependencyGraph` for each of the given targets.

        :param target_names:    The names of the targets for which graphs should be created
        :param clean:           True, if the targets should be cleaned, False otherwise
        :return:                A list that contains the graphs that have been created
        """
        if not target_names:
            Log.error('No targets given')

        invalid_targets = [target_name for target_name in target_names if target_name not in self.targets_by_name]

        if invalid_targets:
            Log.error('The following targets are invalid: %s', format_iterable(invalid_targets))

        return DependencyGraph(self.targets_by_name, *target_names, clean=clean)
