"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for importing and executing Python code at runtime.
"""
import sys

from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType as Module


def import_source_file(source_file: str) -> Module:
    """
    Imports a given source file.

    :param source_file: The path to the source file
    :return:            The module that has been imported
    """
    try:
        spec = spec_from_file_location(source_file, source_file)
        module = module_from_spec(spec)
        sys.modules[source_file] = module
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError as error:
        raise ImportError('Source file "' + source_file + '" not found') from error
