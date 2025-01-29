"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for updating the project's GitHub Actions.
"""
from targets.dependencies.github.modules import GithubWorkflowModule
from targets.paths import Project

MODULES = [
    GithubWorkflowModule(root_directory=Project.Github.root_directory),
]
