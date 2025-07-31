"""
Defines build targets for updating the project's changelog.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.changelog.changelog import print_latest_changelog, update_changelog_bugfix, update_changelog_feature, \
    update_changelog_main, validate_changelog_bugfix, validate_changelog_feature, validate_changelog_main

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target('validate_changelog_bugfix').set_functions(validate_changelog_bugfix) \
    .add_phony_target('validate_changelog_feature').set_functions(validate_changelog_feature) \
    .add_phony_target('validate_changelog_main').set_functions(validate_changelog_main) \
    .add_phony_target('update_changelog_bugfix').set_functions(update_changelog_bugfix) \
    .add_phony_target('update_changelog_feature').set_functions(update_changelog_feature) \
    .add_phony_target('update_changelog_main').set_functions(update_changelog_main) \
    .add_phony_target('print_latest_changelog').set_functions(print_latest_changelog) \
    .build()
