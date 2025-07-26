"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for updating the project's GitHub runners.
"""
from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.dependencies.github.runners.targets import CheckGithubRunners, UpdateGithubRunners

TARGETS = TargetBuilder(BuildUnit.for_file(Path(__file__))) \
    .add_phony_target('check_github_runners').set_runnables(CheckGithubRunners()) \
    .add_phony_target('update_github_runners').set_runnables(UpdateGithubRunners()) \
    .build()
