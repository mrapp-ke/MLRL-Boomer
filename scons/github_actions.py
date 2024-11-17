"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking the project's GitHub workflows for outdated Actions.
"""
import sys

from dataclasses import dataclass, field
from functools import reduce
from glob import glob
from os import environ, path
from typing import List, Optional, Set

from dependencies import install_build_dependencies
from environment import get_env

ENV_GITHUB_TOKEN = 'GITHUB_TOKEN'

WORKFLOW_ENCODING = 'utf-8'


@dataclass
class ActionVersion:
    """
    The version of a GitHub Action.

    Attributes:
        version: The full version string
    """
    version: str

    SEPARATOR = '.'

    @staticmethod
    def from_version_numbers(*version_numbers: int) -> 'ActionVersion':
        """
        Creates and returns the version of a GitHub Action from one or several version numbers.

        :param version_numbers: The version numbers
        :return:                The version that has been created
        """
        return ActionVersion(ActionVersion.SEPARATOR.join([str(version_number) for version_number in version_numbers]))

    @property
    def version_numbers(self) -> List[int]:
        """
        A list that stores the individual version numbers, the full version consists of.
        """
        return [int(version_number) for version_number in str(self).split(self.SEPARATOR)]

    def __str__(self) -> str:
        return self.version.lstrip('v')

    def __lt__(self, other: 'ActionVersion') -> bool:
        first_version_numbers = self.version_numbers
        second_version_numbers = other.version_numbers

        for i in range(min(len(first_version_numbers), len(second_version_numbers))):
            first_version_number = first_version_numbers[i]
            second_version_number = second_version_numbers[i]

            if first_version_number > second_version_number:
                return False
            if first_version_number < second_version_number:
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

    SEPARATOR = '@'

    @staticmethod
    def from_uses_clause(uses_clause: str) -> 'Action':
        """
        Creates and returns a GitHub Action from the uses-clause of a workflow.

        :param uses_clause: The uses-clause
        :return:            The GitHub Action that has been created
        """
        parts = uses_clause.split(Action.SEPARATOR)

        if len(parts) != 2:
            raise ValueError('Uses-clause must contain the symbol + "' + Action.SEPARATOR + '", but got "' + uses_clause
                             + '"')

        return Action(name=parts[0], version=ActionVersion(parts[1]))

    @property
    def repository(self) -> str:
        """
        The name of the repository, where the GitHub Action is hosted.
        """
        repository = self.name
        separator = '/'
        parts = repository.split(separator)
        return separator.join(parts[:2]) if len(parts) > 2 else repository

    @property
    def is_outdated(self) -> bool:
        """
        True, if the GitHub Action is known to be outdated, False otherwise.
        """
        return self.latest_version and self.version < self.latest_version

    def __str__(self) -> str:
        return self.name + self.SEPARATOR + str(self.version)

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
        yaml_dict:      A dictionary that stores the YAML structure of the workflow definition file
        actions:        A set that stores all Actions in the workflow
    """
    workflow_file: str
    yaml_dict: dict
    actions: Set[Action] = field(default_factory=set)

    TAG_USES = 'uses'

    @property
    def uses_clauses(self) -> List[str]:
        """
        A list that contains all uses-clauses in the workflow.
        """
        uses_clauses = []

        for job in self.yaml_dict.get('jobs', {}).values():
            for step in job.get('steps', []):
                uses_clause = step.get(self.TAG_USES, None)

                if uses_clause:
                    uses_clauses.append(uses_clause)

        return uses_clauses

    @property
    def outdated_actions(self) -> Set[Action]:
        """
        A set that stores all Actions in the workflow that are known to be outdated.
        """
        return {action for action in self.actions if action.is_outdated}

    def __eq__(self, other: 'Workflow') -> bool:
        return self.workflow_file == other.workflow_file

    def __hash__(self):
        return hash(self.workflow_file)


def __read_workflow(workflow_file: str) -> Workflow:
    install_build_dependencies('pyyaml')
    # pylint: disable=import-outside-toplevel
    import yaml
    with open(workflow_file, mode='r', encoding=WORKFLOW_ENCODING) as file:
        yaml_dict = yaml.load(file.read(), Loader=yaml.CLoader)
        return Workflow(workflow_file=workflow_file, yaml_dict=yaml_dict)


def __read_workflow_lines(workflow_file: str) -> List[str]:
    with open(workflow_file, mode='r', encoding=WORKFLOW_ENCODING) as file:
        return file.readlines()


def __write_workflow_lines(workflow_file: str, lines: List[str]):
    with open(workflow_file, mode='w', encoding=WORKFLOW_ENCODING) as file:
        file.writelines(lines)


def __update_workflow(workflow_file: str, *updated_actions: Action):
    updated_actions_by_name = reduce(lambda aggr, x: dict(aggr, **{x.name: x}), updated_actions, {})
    lines = __read_workflow_lines(workflow_file)
    uses_prefix = Workflow.TAG_USES + ':'
    updated_lines = []

    for line in lines:
        updated_lines.append(line)
        line_stripped = line.strip()

        if line_stripped.startswith(uses_prefix):
            uses_clause = line_stripped[len(uses_prefix):].strip()
            action = Action.from_uses_clause(uses_clause)
            updated_action = updated_actions_by_name.get(action.name)

            if updated_action:
                updated_lines[-1] = line.replace(str(action.version), str(updated_action.version))

    __write_workflow_lines(workflow_file, updated_lines)


def __parse_workflow(workflow_file: str) -> Workflow:
    print('Searching for GitHub Actions in workflow "' + workflow_file + '"...')
    workflow = __read_workflow(workflow_file)

    for uses_clause in workflow.uses_clauses:
        try:
            workflow.actions.add(Action.from_uses_clause(uses_clause))
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


def __parse_all_workflows() -> Set[Workflow]:
    workflow_directory = path.join('.github', 'workflows')
    workflow_files = glob(path.join(workflow_directory, '*.y*ml'))
    return __determine_latest_action_versions(*__parse_workflows(*workflow_files))


def __print_table(header: List[str], rows: List[List[str]]):
    install_build_dependencies('tabulate')
    # pylint: disable=import-outside-toplevel
    from tabulate import tabulate
    print(tabulate(rows, headers=header))


def __print_outdated_actions(*workflows: Workflow):
    rows = []

    for workflow in workflows:
        for action in workflow.outdated_actions:
            rows.append([workflow.workflow_file, str(action.name), str(action.version), str(action.latest_version)])

    if rows:
        rows.sort(key=lambda row: (row[0], row[1]))
        header = ['Workflow', 'Action', 'Current version', 'Latest version']
        print('The following GitHub Actions are outdated:\n')
        __print_table(header=header, rows=rows)
    else:
        print('All GitHub Actions are up-to-date!')


def __update_outdated_actions(*workflows: Workflow) -> Set[Workflow]:
    rows = []

    for workflow in workflows:
        outdated_actions = workflow.outdated_actions

        if outdated_actions:
            workflow_file = workflow.workflow_file
            updated_actions = set()

            for action in outdated_actions:
                previous_version = action.version
                previous_version_numbers = previous_version.version_numbers
                latest_version_numbers = action.latest_version.version_numbers
                max_version_numbers = min(len(previous_version_numbers), len(latest_version_numbers))
                updated_version = ActionVersion.from_version_numbers(*latest_version_numbers[:max_version_numbers])
                rows.append([workflow_file, action.name, str(previous_version), str(updated_version)])
                action.version = updated_version
                updated_actions.add(action)

            __update_workflow(workflow_file, *updated_actions)

    if rows:
        rows.sort(key=lambda row: (row[0], row[1]))
        header = ['Workflow', 'Action', 'Previous version', 'Updated version']
        print('The following GitHub Actions have been updated:\n')
        __print_table(header=header, rows=rows)
    else:
        print('No GitHub Actions have been updated.')

    return set(workflows)


def check_github_actions(**_):
    """
    Checks the project's GitHub workflows for outdated Actions.
    """
    workflows = __parse_all_workflows()
    __print_outdated_actions(*workflows)


def update_github_actions(**_):
    """
    Updates the versions of outdated GitHub Actions in the project's workflows.
    """
    workflows = __parse_all_workflows()
    __update_outdated_actions(*workflows)
