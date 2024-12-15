"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Initializes the build system and runs targets specified via command line arguments.
"""
import sys

from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import List, Optional

from core.build_unit import BuildUnit
from core.modules import Module, ModuleRegistry
from core.targets import DependencyGraph, Target, TargetRegistry
from util.files import FileSearch
from util.format import format_iterable
from util.log import Log


def __parse_command_line_arguments():
    parser = ArgumentParser(description='The build system of the project "MLRL-Boomer"')
    parser.add_argument('--verbose', action='store_true', help='Enables verbose logging.')
    parser.add_argument('--clean', action='store_true', help='Cleans the specified targets.')
    parser.add_argument('targets', nargs='*')
    return parser.parse_args()


def __configure_log(args):
    log_level = Log.Level.VERBOSE if args.verbose else Log.Level.INFO
    Log.configure(log_level)


def __find_init_files() -> List[str]:
    return FileSearch() \
        .set_recursive(True) \
        .filter_by_name('__init__.py') \
        .list(BuildUnit().root_directory)


def __import_source_file(source_file: str) -> ModuleType:
    try:
        spec = spec_from_file_location(source_file, source_file)
        module = module_from_spec(spec)
        sys.modules[source_file] = module
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError as error:
        raise ImportError('Source file "' + source_file + '" not found') from error


def __register_modules(init_files: List[str]) -> ModuleRegistry:
    Log.verbose('Registering modules...')
    module_registry = ModuleRegistry()
    num_modules = 0

    for init_file in init_files:
        modules = [
            module for module in getattr(__import_source_file(init_file), 'MODULES', []) if isinstance(module, Module)
        ]

        if modules:
            Log.verbose('Registering %s modules defined in file "%s":\n', str(len(modules)), init_file)

            for module in modules:
                Log.verbose(' - %s', str(module))
                module_registry.register(module)

            Log.verbose('')
            num_modules += len(modules)

    Log.verbose('Successfully registered %s modules.\n', str(num_modules))
    return module_registry


def __register_targets(init_files: List[str]) -> TargetRegistry:
    Log.verbose('Registering targets...')
    target_registry = TargetRegistry()
    num_targets = 0

    for init_file in init_files:
        targets = [
            target for target in getattr(__import_source_file(init_file), 'TARGETS', []) if isinstance(target, Target)
        ]

        if targets:
            Log.verbose('Registering %s targets defined in file "%s":\n', str(len(targets)), init_file)

            for target in targets:
                Log.verbose(' - %s', str(target))
                target_registry.register(target)

            Log.verbose('')
            num_targets += len(targets)

    Log.verbose('Successfully registered %s targets.\n', str(num_targets))
    return target_registry


def __find_default_target(init_files: List[str]) -> Optional[str]:
    Log.verbose('Searching for default target...')
    default_targets = []

    for init_file in init_files:
        default_target = getattr(__import_source_file(init_file), 'DEFAULT_TARGET', None)

        if default_target and isinstance(default_target, str):
            Log.verbose('Found default target "%s" defined in file "%s"', default_target, init_file)
            default_targets.append(default_target)

    if len(default_targets) > 1:
        raise RuntimeError('Only one default target may be specified, but found: ' + format_iterable(default_targets))

    if default_targets:
        return default_targets[0]

    Log.verbose('Found no default target.')
    return None


def __create_dependency_graph(target_registry: TargetRegistry, args, default_target: Optional[str]) -> DependencyGraph:
    targets = args.targets
    clean = args.clean
    graph_type = DependencyGraph.Type.CLEAN if clean else DependencyGraph.Type.RUN

    if not targets and default_target:
        targets = [default_target]

        if clean:
            graph_type = DependencyGraph.Type.CLEAN_ALL

    Log.verbose('Creating dependency graph for %s targets [%s]...', 'cleaning' if clean else 'running',
                format_iterable(targets))
    dependency_graph = target_registry.create_dependency_graph(*targets, graph_type=graph_type)
    Log.verbose('Successfully created dependency graph:\n\n%s\n', str(dependency_graph))
    return dependency_graph


def __execute_dependency_graph(dependency_graph: DependencyGraph, module_registry: ModuleRegistry):
    Log.verbose('Executing dependency graph...')
    dependency_graph.execute(module_registry)
    Log.verbose('Successfully executed dependency graph.')


def main():
    """
    The main function to be executed when the build system is invoked.
    """
    args = __parse_command_line_arguments()
    __configure_log(args)

    init_files = __find_init_files()
    module_registry = __register_modules(init_files)
    target_registry = __register_targets(init_files)
    default_target = __find_default_target(init_files)
    dependency_graph = __create_dependency_graph(target_registry, args, default_target=default_target)
    __execute_dependency_graph(dependency_graph, module_registry)


if __name__ == '__main__':
    main()
