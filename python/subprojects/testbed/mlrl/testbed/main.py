"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Imports and invokes the program to be run by the command line utility.
"""
import logging as log
import sys

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from enum import Enum
from importlib import import_module
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from typing import Optional

from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode
from mlrl.testbed.program_info import ProgramInfo
from mlrl.testbed.runnables import Runnable

from mlrl.util.cli import CommandLineInterface, EnumArgument


class LogLevel(Enum):
    """
    Specifies all valid textual representations of log levels.
    """
    DEBUG = log.DEBUG
    INFO = log.INFO
    WARN = log.WARN
    WARNING = log.WARNING
    ERROR = log.ERROR
    CRITICAL = log.CRITICAL
    FATAL = log.FATAL
    NOTSET = log.NOTSET


LOG_LEVEL = EnumArgument(
    '--log-level',
    enum=LogLevel,
    default=LogLevel.INFO,
    description='The log level to be used.',
)


def __create_argument_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='A command line utility for training and evaluating machine learning algorithms',
        add_help=False)

    argument_parser.add_argument('runnable_module_or_source_file',
                                 nargs='?' if '--version' in sys.argv or '-v' in sys.argv else None,
                                 default=None,
                                 type=str,
                                 help='The Python module or source file of the program that should be run')
    argument_parser.add_argument('-r',
                                 '--runnable',
                                 default='Runnable',
                                 type=str,
                                 help='The Python class name of the program that should be run')
    return argument_parser


def __get_runnable(argument_parser: ArgumentParser) -> Optional[Runnable]:
    args = vars(argument_parser.parse_known_args()[0])
    runnable_module_or_source_file = args.get('runnable_module_or_source_file', None)
    runnable_class_name = args.get('runnable', None)
    runnable = None

    if runnable_module_or_source_file and runnable_class_name:
        runnable = __instantiate_via_default_constructor(module_or_source_file=str(runnable_module_or_source_file),
                                                         class_name=str(runnable_class_name))

        if not isinstance(runnable, Runnable):
            raise TypeError('Class "' + str(runnable_class_name) + '" must extend from "' + Runnable.__qualname__ + '"')

    return runnable


def __import_module(module_name: str):
    try:
        return import_module(module_name)
    except ImportError as error:
        raise ImportError('Module "' + module_name + '" not found') from error


def __import_source_file(source_file: str):
    try:
        module_name = 'runnable'
        spec = spec_from_file_location(module_name, source_file)

        if spec:
            spec_loader = spec.loader

            if spec_loader:
                module = module_from_spec(spec)
                sys.modules[module_name] = module
                spec_loader.exec_module(module)
                return module

        raise FileNotFoundError()
    except FileNotFoundError as error:
        raise ImportError('Source file "' + source_file + '" not found') from error


def __import_module_or_source_file(module_name: str):
    try:
        return __import_module(module_name)
    except ImportError:
        try:
            return __import_source_file(module_name)
        except ImportError:
            # pylint: disable=raise-missing-from
            raise ImportError('Failed to import module or source file')


def __import_class(module_or_source_file: str, class_name: str):
    try:
        module = __import_module_or_source_file(module_or_source_file)
        return getattr(module, class_name)
    except AttributeError as error:
        raise ImportError('Class "' + class_name + '" not found') from error


def __instantiate_via_default_constructor(module_or_source_file: str, class_name: str):
    try:
        class_type = __import_class(module_or_source_file, class_name)
        return class_type()
    except TypeError as error:
        raise TypeError('Class "' + class_name + '" must provide a default constructor') from error


def __get_cli(runnable: Optional[Runnable], argument_parser: ArgumentParser) -> CommandLineInterface:
    program_info = runnable.get_program_info() if runnable else __get_default_program_info()
    cli = CommandLineInterface(argument_parser, version_text=str(program_info) if program_info else None)
    cli.add_arguments(LOG_LEVEL)
    return cli


def __get_default_program_info() -> ProgramInfo:
    package_name = 'mlrl-testbed'
    return ProgramInfo(
        name=package_name,
        version=version(package_name),
        year='2020 - 2025',
        authors=['Michael Rapp et al.'],
    )


def __get_mode(cli: CommandLineInterface, runnable: Optional[Runnable]) -> Mode:
    cli.add_arguments(Mode.MODE)
    args = cli.parse_known_args()
    mode = Mode.MODE.get_value(args)

    if mode == ExperimentMode.BATCH:
        return runnable.configure_batch_mode(cli) if runnable else BatchMode()
    if mode == ExperimentMode.READ:
        return runnable.configure_read_mode(cli) if runnable else ReadMode()
    if mode == ExperimentMode.RUN:
        return runnable.configure_run_mode(cli) if runnable else RunMode()
    return SingleMode()


def __configure_logger(args: Namespace):
    log_level = LOG_LEVEL.get_value(args).value
    root = log.getLogger()
    root.setLevel(log_level)
    out_handler = log.StreamHandler(sys.stdout)
    out_handler.setLevel(log_level)
    out_handler.setFormatter(log.Formatter('%(message)s'))
    existing_handlers = list(root.handlers)

    for existing_handler in existing_handlers:
        root.removeHandler(existing_handler)

    root.addHandler(out_handler)


def main():
    """
    The main function to be executed when the program starts.
    """
    argument_parser = __create_argument_parser()
    runnable = __get_runnable(argument_parser)
    cli = __get_cli(runnable, argument_parser)
    mode = __get_mode(cli, runnable)

    if runnable:
        runnable.configure_arguments(cli, mode)
    else:
        mode.configure_arguments(cli, extension_arguments=[], algorithmic_arguments=[])

    argument_parser.add_argument('-h',
                                 '--help',
                                 action='help',
                                 default='==SUPPRESS==',
                                 help='Show this help message and exit')

    args = argument_parser.parse_args()
    __configure_logger(args)

    if runnable:
        runnable.run(mode, cli.arguments, args)


if __name__ == '__main__':
    main()
