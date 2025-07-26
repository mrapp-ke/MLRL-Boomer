"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines modules that provide access to GitHub workflows.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.dependencies.github.actions.targets import CheckGithubActions, UpdateGithubActions

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target('check_github_actions').set_runnables(CheckGithubActions()) \
    .add_phony_target('update_github_actions').set_runnables(UpdateGithubActions()) \
    .build()
