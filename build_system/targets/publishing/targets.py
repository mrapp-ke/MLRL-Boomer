"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for publishing pre-built packages.
"""
from os import environ
from pathlib import Path

from core.build_unit import BuildUnit
from util.env import get_env
from util.format import format_iterable

from targets.publishing.cibuildwheel import Cibuildwheel

ENV_PACKAGE_DIRECTORY = 'CIBW_PACKAGE_DIR'


def print_cibuildwheel_identifiers(build_unit: BuildUnit):
    """
    Prints cibuildwheel build identifiers as a JSON array.

    :param build_unit: The build unit, the target belongs to
    """
    package_directory = get_env(environ, ENV_PACKAGE_DIRECTORY)

    if not package_directory:
        raise ValueError('A directory must be specified via the environment variable ' + ENV_PACKAGE_DIRECTORY)

    stdout = Cibuildwheel(build_unit, package_directory=Path(package_directory), print_build_identifiers=True) \
        .install_program(True, silent=True) \
        .print_command(False) \
        .capture_output()
    build_identifiers = set(filter(None, map(lambda build_identifier: build_identifier.strip(), stdout.split('\n'))))
    print('[' + format_iterable(sorted(build_identifiers), delimiter='"') + ']')
