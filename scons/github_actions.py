"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking the project's GitHub workflows for outdated Actions.
"""
import sys

from dataclasses import dataclass, field
from glob import glob
from os import environ, path
from typing import List, Optional, Set

from dependencies import install_build_dependencies
from environment import get_env

ENV_GITHUB_TOKEN = 'GITHUB_TOKEN'

SEPARATOR_VERSION = '@'

SEPARATOR_VERSION_NUMBER = '.'

SEPARATOR_PATH = '/'


@dataclass
class ActionVersion:
    """
    The version of a GitHub Action.

    Attributes:
        version: The full version string
    """
    version: str

    def __str__(self) -> str:
        return self.version.lstrip('v')

    def __lt__(self, other: 'ActionVersion') -> bool:
        first_numbers = str(self).split(SEPARATOR_VERSION_NUMBER)
        second_numbers = str(other).split(SEPARATOR_VERSION_NUMBER)

        for i in range(min(len(first_numbers), len(second_numbers))):
            first = int(first_numbers[i])
            second = int(second_numbers[i])

            if first > second:
                return False
            if first < second:
                return True

        return False


@dataclass
class Action:
    """
    A GitHub Action.

    Attributes:
        name:           The name of the Action
        version:        The version of the Action
        latest_version: The latest version of the Action, if known
    """
    name: str
    version: ActionVersion
    latest_version: Optional[ActionVersion] = None

    @staticmethod
    def parse(uses: str) -> 'Action':
        """
        Parses and returns a GitHub Action as specified via the uses-clause of a workflow.

        :param uses:    The uses-clause
        :return:        The GitHub Action
        """
        parts = uses.split(SEPARATOR_VERSION)

        if len(parts) != 2:
            raise ValueError('Action must contain the symbol + "' + SEPARATOR_VERSION + '", but got "' + uses + '"')

        return Action(name=parts[0], version=ActionVersion(parts[1]))

    @property
    def repository(self) -> str:
        """
        The name of the repository, where the GitHub Action is hosted.
        """
        repository = self.name
        parts = repository.split(SEPARATOR_PATH)
        return SEPARATOR_PATH.join(parts[:2]) if len(parts) > 2 else repository

    def is_outdated(self) -> bool:
        """
        Returns whether the GitHub Action is known to be outdated or not.

        :return: True, if the GitHub Action is outdated, False otherwise
        """
        return self.latest_version and self.version < self.latest_version

    def __str__(self) -> str:
        return self.name + SEPARATOR_VERSION + str(self.version)

    def __eq__(self, other: 'Action') -> bool:
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


@dataclass
class Workflow:
    """
    A GitHub workflow.

    Attributes:
        workflow_file:  The path of the workflow definition file
        actions:        A set that stores the Actions in the workflow
    """
    workflow_file: str
    actions: Set[Action] = field(default_factory=set)

    def __eq__(self, other: 'Workflow') -> bool:
        return self.workflow_file == other.workflow_file

    def __hash__(self):
        return hash(self.workflow_file)


def __get_github_workflow_files(directory: str) -> List[str]:
    return glob(path.join(directory, '*.y*ml'))


def __load_yaml(workflow_file: str) -> dict:
    install_build_dependencies('pyyaml')
    # pylint: disable=import-outside-toplevel
    import yaml
    with open(workflow_file, encoding='utf-8') as file:
        return yaml.load(file.read(), Loader=yaml.CLoader)


def __parse_workflow(workflow_file: str) -> Workflow:
    print('Searching for GitHub Actions in workflow "' + workflow_file + '"...')
    workflow = Workflow(workflow_file)
    workflow_yaml = __load_yaml(workflow_file)

    for job in workflow_yaml.get('jobs', {}).values():
        for step in job.get('steps', []):
            uses = step.get('uses', None)

            if uses:
                try:
                    action = Action.parse(uses)
                    workflow.actions.add(action)
                except ValueError as error:
                    print('Failed to parse uses-clause in workflow "' + workflow_file + '": ' + str(error))
                    sys.exit(-1)

    return workflow


def __parse_workflows(*workflow_files: str) -> Set[Workflow]:
    return {__parse_workflow(workflow_file) for workflow_file in workflow_files}


def __query_latest_action_version(action: Action, github_token: Optional[str] = None) -> Optional[ActionVersion]:
    repository_name = action.repository
    install_build_dependencies('pygithub')
    # pylint: disable=import-outside-toplevel
    from github import Auth, Github, UnknownObjectException

    try:
        github_auth = Auth.Token(github_token) if github_token else None
        github_client = Github(auth=github_auth)
        github_repository = github_client.get_repo(repository_name)
        latest_release = github_repository.get_latest_release()
        latest_tag = latest_release.tag_name
        return ActionVersion(latest_tag)
    except UnknownObjectException as error:
        print('Query to GitHub API failed for action "' + str(action) + '" hosted in repository "' + repository_name
              + '": ' + str(error))
        sys.exit(-1)


def __get_github_token() -> Optional[str]:
    github_token = get_env(environ, ENV_GITHUB_TOKEN)

    if not github_token:
        print('No GitHub API token is set. You can specify it via the environment variable ' + ENV_GITHUB_TOKEN + '.')

    return github_token


def __determine_latest_action_versions(*workflows: Workflow) -> Set[Workflow]:
    github_token = __get_github_token()
    version_cache = {}

    for workflow in workflows:
        for action in workflow.actions:
            latest_version = version_cache.get(action.name)

            if not latest_version:
                print('Checking version of GitHub Action "' + action.name + '"...')
                latest_version = __query_latest_action_version(action, github_token=github_token)
                version_cache[action.name] = latest_version

            action.latest_version = latest_version

    return set(workflows)


def __print_outdated_actions(*workflows: Workflow):
    rows = []

    for workflow in workflows:
        for action in workflow.actions:
            if action.is_outdated():
                rows.append([workflow.workflow_file, str(action.name), str(action.version), str(action.latest_version)])

    if rows:
        rows.sort(key=lambda row: (row[0], row[1]))
        header = ['Workflow', 'Action', 'Current version', 'Latest version']
        install_build_dependencies('tabulate')
        # pylint: disable=import-outside-toplevel
        from tabulate import tabulate
        print('The following GitHub Actions are outdated:\n')
        print(tabulate(rows, headers=header))


def check_github_actions(**_):
    """
    Checks the project's GitHub workflows for outdated Actions.
    """
    workflow_directory = path.join('.github', 'workflows')
    workflow_files = __get_github_workflow_files(workflow_directory)
    workflows = __determine_latest_action_versions(*__parse_workflows(*workflow_files))
    __print_outdated_actions(*workflows)


def update_github_actions(**_):
    """
    Updates the versions of outdated GitHub Actions in the project's workflows.
    """
    print('Updating versions of outdated GitHub Actions...')
