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
    PhonyTarget('increment_development_version', action=increment_development_version),
    PhonyTarget('reset_development_version', action=reset_development_version),
    PhonyTarget('apply_development_version', action=apply_development_version),
    PhonyTarget('increment_patch_version', action=increment_patch_version),
    PhonyTarget('increment_minor_version', action=increment_minor_version),
    PhonyTarget('increment_major_version', action=increment_major_version),

    # Targets for validating changelogs
    PhonyTarget('validate_changelog_bugfix', action=validate_changelog_bugfix),
    PhonyTarget('validate_changelog_feature', action=validate_changelog_feature),
    PhonyTarget('validate_changelog_main', action=validate_changelog_main),

    # Targets for updating the project's changelog
    PhonyTarget('update_changelog_bugfix', action=update_changelog_bugfix),
    PhonyTarget('update_changelog_feature', action=update_changelog_feature),
    PhonyTarget('update_changelog_main', action=update_changelog_main),

    # Targets for printing information about the project
    PhonyTarget(name='print_version', action=print_current_version),
    PhonyTarget(name='print_latest_changelog', action=print_latest_changelog)
]
