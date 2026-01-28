"""
Defines build targets for publishing pre-built packages.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.publishing.targets import print_cibuildwheel_identifiers, print_cibuildwheel_python_versions

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target('print_cibuildwheel_identifiers').set_functions(print_cibuildwheel_identifiers) \
    .add_phony_target('print_cibuildwheel_python_versions').set_functions(print_cibuildwheel_python_versions) \
    .build()
