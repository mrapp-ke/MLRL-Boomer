"""
Defines build targets for triggering readthedocs builds.
"""
from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.documentation.rdt.targets import trigger_readthedocs_build

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target('trigger_readthedocs_build').set_functions(trigger_readthedocs_build) \
    .build()
