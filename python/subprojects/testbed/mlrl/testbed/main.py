"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Imports and invokes the program to be run by the command line utility.
"""
import sys

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location

from mlrl.testbed.runnables import Runnable


def __create_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='A command line utility for training and evaluating machine learning algorithms',
        add_help=False)
    parser.add_argument('runnable_module_or_source_file',
                        type=str,
                        help='The Python module or source file of the program that should be run')
    parser.add_argument('-r',
                        '--runnable',
                        default='Runnable',
                        type=str,
                        help='The Python class name of the program that should be run')
    return parser


def __import_module(module_name: str):
    try:
        return import_module(module_name)
    except ImportError as error:
        raise ImportError('Module "' + module_name + '" not found') from error


def __import_source_file(source_file: str):
    try:
        module_name = 'runnable'
        spec = spec_from_file_location(module_name, source_file)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
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


def __configure_arguments(runnable: Runnable, parser: ArgumentParser):
    runnable.configure_arguments(parser)
    parser.add_argument('-h', '--help', action='help', default='==SUPPRESS==', help='Show this help message and exit')


def main():
    """
    The main function to be executed when the program starts.
    """
    parser = __create_argument_parser()
    args = parser.parse_known_args()[0]

    runnable_module_or_source_file = str(args.runnable_module_or_source_file)
    runnable_class_name = str(args.runnable)
    runnable = __instantiate_via_default_constructor(module_or_source_file=runnable_module_or_source_file,
                                                     class_name=runnable_class_name)

    if not isinstance(runnable, Runnable):
        raise TypeError('Class "' + runnable_class_name + '" must extend from "' + Runnable.__qualname__ + '"')

    __configure_arguments(runnable, parser)
    runnable.run(parser.parse_args())


if __name__ == '__main__':
    main()
