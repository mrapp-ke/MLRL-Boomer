"""
Defines build targets for updating the project's version and changelog.
"""
from util.targets import PhonyTarget
from versioning.changelog import print_latest_changelog, update_changelog_bugfix, update_changelog_feature, \
    update_changelog_main, validate_changelog_bugfix, validate_changelog_feature, validate_changelog_main
from versioning.versioning import apply_development_version, increment_development_version, increment_major_version, \
    increment_minor_version, increment_patch_version, print_current_version, reset_development_version

TARGETS = [
    # Targets for updating the project's version
    PhonyTarget.Builder('increment_development_version').set_function(increment_development_version).build(),
    PhonyTarget.Builder('reset_development_version').set_function(reset_development_version).build(),
    PhonyTarget.Builder('apply_development_version').set_function(apply_development_version).build(),
    PhonyTarget.Builder('increment_patch_version').set_function(increment_patch_version).build(),
    PhonyTarget.Builder('increment_minor_version').set_function(increment_minor_version).build(),
    PhonyTarget.Builder('increment_major_version').set_function(increment_major_version).build(),

    # Targets for validating changelogs
    PhonyTarget.Builder('validate_changelog_bugfix').set_function(validate_changelog_bugfix).build(),
    PhonyTarget.Builder('validate_changelog_feature').set_function(validate_changelog_feature).build(),
    PhonyTarget.Builder('validate_changelog_main').set_function(validate_changelog_main).build(),

    # Targets for updating the project's changelog
    PhonyTarget.Builder('update_changelog_bugfix').set_function(update_changelog_bugfix).build(),
    PhonyTarget.Builder('update_changelog_feature').set_function(update_changelog_feature).build(),
    PhonyTarget.Builder('update_changelog_main').set_function(update_changelog_main).build(),

    # Targets for printing information about the project
    PhonyTarget.Builder('print_version').set_function(print_current_version).build(),
    PhonyTarget.Builder('print_latest_changelog').set_function(print_latest_changelog).build()
]
