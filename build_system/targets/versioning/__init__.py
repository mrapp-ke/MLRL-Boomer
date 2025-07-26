"""
Defines build targets for updating the project's version.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.versioning.versioning import increment_development_version, increment_major_version, \
    increment_minor_version, increment_patch_version, reset_development_version

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target('increment_development_version').set_functions(increment_development_version) \
    .add_phony_target('reset_development_version').set_functions(reset_development_version) \
    .add_phony_target('increment_patch_version').set_functions(increment_patch_version) \
    .add_phony_target('increment_minor_version').set_functions(increment_minor_version) \
    .add_phony_target('increment_major_version').set_functions(increment_major_version) \
    .build()
