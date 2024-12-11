"""
Defines build targets for updating the project's version and changelog.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder
from versioning.changelog import print_latest_changelog, update_changelog_bugfix, update_changelog_feature, \
    update_changelog_main, validate_changelog_bugfix, validate_changelog_feature, validate_changelog_main
from versioning.versioning import apply_development_version, increment_development_version, increment_major_version, \
    increment_minor_version, increment_patch_version, print_current_version, reset_development_version

TARGETS = TargetBuilder(BuildUnit('util')) \
    .add_phony_target('increment_development_version').set_functions(increment_development_version) \
    .add_phony_target('reset_development_version').set_functions(reset_development_version) \
    .add_phony_target('apply_development_version').set_functions(apply_development_version) \
    .add_phony_target('increment_patch_version').set_functions(increment_patch_version) \
    .add_phony_target('increment_minor_version').set_functions(increment_minor_version) \
    .add_phony_target('increment_major_version').set_functions(increment_major_version) \
    .add_phony_target('validate_changelog_bugfix').set_functions(validate_changelog_bugfix) \
    .add_phony_target('validate_changelog_feature').set_functions(validate_changelog_feature) \
    .add_phony_target('validate_changelog_main').set_functions(validate_changelog_main) \
    .add_phony_target('update_changelog_bugfix').set_functions(update_changelog_bugfix) \
    .add_phony_target('update_changelog_feature').set_functions(update_changelog_feature) \
    .add_phony_target('update_changelog_main').set_functions(update_changelog_main) \
    .add_phony_target('print_version').set_functions(print_current_version) \
    .add_phony_target('print_latest_changelog').set_functions(print_latest_changelog) \
    .build()
