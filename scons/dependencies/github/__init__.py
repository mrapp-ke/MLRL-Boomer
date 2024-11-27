"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for updating the project's GitHub Actions.
"""
from dependencies.github.targets import CheckGithubActions, UpdateGithubActions
from util.targets import PhonyTarget, TargetBuilder
from util.units import BuildUnit

TARGETS = TargetBuilder(BuildUnit('dependencies', 'github')) \
    .add_phony_target('check_github_actions').set_runnables(CheckGithubActions()) \
    .add_phony_target('update_github_actions').set_runnables(UpdateGithubActions()) \
    .build()
