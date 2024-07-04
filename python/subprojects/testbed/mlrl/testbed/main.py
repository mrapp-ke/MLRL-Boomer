"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Imports and invokes the program to be run by the command line utility.
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from importlib import import_module

from mlrl.testbed.runnables import Runnable


def __create_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='A command line utility for training and evaluating machine learning algorithms')
    parser.add_argument('runnable_module', type=str, help='The Python module of the program that should be run')
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


def __import_class(module_name: str, class_name: str):
    try:
        module = __import_module(module_name)
        return getattr(module, class_name)
    except AttributeError as error:
        raise ImportError('Class "' + module_name + '.' + class_name + '" not found') from error


def __instantiate_via_default_constructor(module_name: str, class_name: str):
    try:
        class_type = __import_class(module_name, class_name)
        return class_type()
    except TypeError as error:
        raise TypeError('Class "' + module_name + '.' + class_name + '" must provide a default constructor') from error


def main():
    """
    The main function to be executed when the program starts.
    """
    parser = __create_argument_parser()
    args = parser.parse_known_args()[0]

    runnable_module_name = str(args.runnable_module)
    runnable_class_name = str(args.runnable)
    runnable = __instantiate_via_default_constructor(module_name=runnable_module_name, class_name=runnable_class_name)

    if not isinstance(runnable, Runnable):
        raise TypeError('Class "' + runnable_module_name + '.' + runnable_class_name + '" must extend class "'
                        + Runnable.__qualname__ + '"')

    runnable.configure_arguments(parser)
    runnable.run(parser.parse_args())


if __name__ == '__main__':
    main()
