"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for publishing pre-built packages.
"""
import re as regex

from os import environ
from pathlib import Path
from typing import Any, Iterable, Set

from core.build_unit import BuildUnit
from util.env import get_env
from util.format import format_iterable

from targets.publishing.cibuildwheel import Cibuildwheel

ENV_PACKAGE_DIRECTORY = 'CIBW_PACKAGE_DIR'


def __get_cibuildwheel_identifiers(build_unit: BuildUnit) -> Set[str]:
    package_directory = get_env(environ, ENV_PACKAGE_DIRECTORY)

    if not package_directory:
        raise ValueError('A directory must be specified via the environment variable ' + ENV_PACKAGE_DIRECTORY)

    stdout = Cibuildwheel(build_unit, package_directory=Path(package_directory), print_build_identifiers=True) \
        .install_program(True, silent=True) \
        .print_command(False) \
        .capture_output()
    return set(filter(None, map(lambda build_identifier: build_identifier.strip(), stdout.split('\n'))))


def __parse_python_version(build_identifier: str) -> str:
    parts = build_identifier.split('-')

    if len(parts) >= 2:
        match = regex.match(r'^(cp|pp)(\d)(\d+)(t?)$', parts[0])

        if match:
            _, major_version, minor_version, suffix = match.groups()
            return f'{major_version}.{minor_version}{suffix}'

    raise ValueError('Failed to parse Python version from build identifier: ' + build_identifier)


def __to_json_array(elements: Iterable[Any]) -> str:
    return '[' + format_iterable(elements, delimiter='"') + ']'


def print_cibuildwheel_identifiers(build_unit: BuildUnit):
    """
    Prints cibuildwheel build identifiers as a JSON array.

    :param build_unit: The build unit, the target belongs to
    """
    print(__to_json_array(sorted(__get_cibuildwheel_identifiers(build_unit))))


def print_cibuildwheel_python_versions(build_unit: BuildUnit):
    """
    Prints Python versions taken from cibuildwheel identifiers as a JSON array.

    :param build_unit: The build unit, the target belongs to
    """
    print(__to_json_array(sorted(set(map(__parse_python_version, __get_cibuildwheel_identifiers(build_unit))))))
