"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for updating the project's GitHub Actions.
"""
from core.build_unit import BuildUnit
from core.targets import PhonyTarget, TargetBuilder

from targets.dependencies.github.modules import GithubWorkflowModule
from targets.dependencies.github.targets import CheckGithubActions, UpdateGithubActions
from targets.paths import Project

TARGETS = TargetBuilder(BuildUnit.for_file(__file__)) \
    .add_phony_target('check_github_actions').set_runnables(CheckGithubActions()) \
    .add_phony_target('update_github_actions').set_runnables(UpdateGithubActions()) \
    .build()

MODULES = [
    GithubWorkflowModule(root_directory=Project.Github.root_directory),
]
