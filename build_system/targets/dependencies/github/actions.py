"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking the project's GitHub workflows for outdated Actions.
"""
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from typing import Dict, List, Set

from core.build_unit import BuildUnit
from util.log import Log

from targets.dependencies.github.modules import GithubWorkflowModule
from targets.dependencies.github.workflows import Workflow, Workflows
from targets.dependencies.pygithub import GithubApi


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
    """
    name: str
    version: ActionVersion

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

    def __str__(self) -> str:
        return self.name + self.SEPARATOR + str(self.version)

    def __eq__(self, other: 'Action') -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class Actions(Workflow):
    """
    Allows to access and update the GitHub Actions in a workflow.
    """

    TAG_USES = 'uses'

    @cached_property
    def uses_clauses(self) -> List[str]:
        """
        A list that contains all uses-clauses in the workflow.
        """
        uses_clauses = []

        for job in self.find_tag(self.yaml_dict, 'jobs', default={}).values():
            for step in self.find_tag(job, 'steps', default=[]):
                uses_clause = self.find_tag(step, self.TAG_USES)

                if uses_clause:
                    uses_clauses.append(uses_clause)

        return uses_clauses

    @cached_property
    def actions(self) -> Set[Action]:
        """
        A set that contains all GitHub Actions used in the workflow.
        """
        actions = set()

        for uses_clause in self.uses_clauses:
            try:
                actions.add(Action.from_uses_clause(uses_clause))
            except ValueError as error:
                raise RuntimeError('Failed to parse uses-clause in workflow "' + self.file + '"') from error

        return actions

    def update_actions(self, *updated_actions: Action):
        """
        Updates given Actions in the workflow definition file.

        :param updated_actions: The actions to be updated
        """
        updated_actions_by_name = reduce(lambda aggr, x: dict(aggr, **{x.name: x}), updated_actions, {})
        uses_prefix = self.TAG_USES + ':'
        updated_lines = []

        for line in self.lines:
            updated_lines.append(line)
            line_stripped = line.strip()

            if line_stripped.startswith(uses_prefix):
                uses_clause = line_stripped[len(uses_prefix):].strip()
                action = Action.from_uses_clause(uses_clause)
                updated_action = updated_actions_by_name.get(action.name)

                if updated_action:
                    updated_lines[-1] = line.replace(str(action.version), str(updated_action.version))

        self.write_lines(*updated_lines)

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.uses_clauses
        except AttributeError:
            pass

        try:
            del self.actions
        except AttributeError:
            pass


class WorkflowUpdater(Workflows):
    """
    Allows checking the versions of GitHub Actions used in multiple workflows and updating outdated ones.
    """

    @dataclass
    class OutdatedAction:
        """
        An outdated GitHub Action.

        Attributes:
            action:         The outdated Action
            latest_version: The latest version of the Action
        """
        action: Action
        latest_version: ActionVersion

        def __str__(self) -> str:
            return str(self.action)

        def __eq__(self, other: 'WorkflowUpdater.OutdatedAction') -> bool:
            return self.action == other.action

        def __hash__(self) -> int:
            return hash(self.action)

    @dataclass
    class UpdatedAction:
        """
        A GitHub Action that has been updated.

        Attributes:
            previous:   The previous Action
            updated:    The updated Action
        """
        previous: 'WorkflowUpdater.OutdatedAction'
        updated: Action

        def __str__(self) -> str:
            return str(self.updated)

        def __eq__(self, other: 'WorkflowUpdater.UpdatedAction') -> bool:
            return self.updated == other.updated

        def __hash__(self) -> int:
            return hash(self.updated)

    def __query_latest_action_version(self, action: Action) -> ActionVersion:
        repository_name = action.repository

        try:
            latest_tag = self.github_api \
                .open_repository(repository_name) \
                .get_latest_release() \
                .tag_name

            if not latest_tag:
                raise RuntimeError('No releases available')

            return ActionVersion(latest_tag)
        except RuntimeError as error:
            raise RuntimeError('Unable to determine latest version of action "' + str(action)
                               + '" hosted in repository "' + repository_name + '"') from error

    def __get_latest_action_version(self, action: Action) -> ActionVersion:
        latest_version = self.version_cache.get(action.name)

        if not latest_version:
            Log.info('Checking version of GitHub Action "%s"...', action.name)
            latest_version = self.__query_latest_action_version(action)
            self.version_cache[action.name] = latest_version

        return latest_version

    def __init__(self, build_unit: BuildUnit, module: GithubWorkflowModule):
        """
        :param build_unit:  The build unit from which workflow definition files should be read
        :param module:      The module, that contains the workflow definition files
        """
        super().__init__(build_unit, module)
        self.version_cache = {}
        self.github_api = GithubApi(build_unit).set_token_from_env()

    def find_outdated_workflows(self) -> Dict[Actions, Set[OutdatedAction]]:
        """
        Finds and returns all workflows with outdated GitHub actions.

        :return: A dictionary that contains for each workflow a set of outdated Actions
        """
        outdated_workflows = {}

        for workflow in self.workflows:
            Log.info('Searching for GitHub Actions in workflow "%s"...', workflow.file)
            workflow = Actions(self.build_unit, file=workflow.file)

            for action in workflow.actions:
                latest_version = self.__get_latest_action_version(action)

                if action.version < latest_version:
                    outdated_actions = outdated_workflows.setdefault(workflow, set())
                    outdated_actions.add(WorkflowUpdater.OutdatedAction(action, latest_version))

        return outdated_workflows

    def update_outdated_workflows(self) -> Dict[Actions, Set[UpdatedAction]]:
        """
        Updates all workflows with outdated GitHub Actions.

        :return: A dictionary that contains for each workflow a set of updated Actions
        """
        updated_workflows = {}

        for workflow, outdated_actions in self.find_outdated_workflows().items():
            updated_actions = set()

            for outdated_action in outdated_actions:
                previous_version = outdated_action.action.version
                previous_version_numbers = previous_version.version_numbers
                latest_version_numbers = outdated_action.latest_version.version_numbers
                max_version_numbers = min(len(previous_version_numbers), len(latest_version_numbers))
                updated_version = ActionVersion.from_version_numbers(*latest_version_numbers[:max_version_numbers])
                updated_actions = updated_workflows.setdefault(workflow, updated_actions)
                updated_action = replace(outdated_action.action, version=updated_version)
                updated_actions.add(WorkflowUpdater.UpdatedAction(previous=outdated_action, updated=updated_action))

            workflow.update_actions(*[updated_action.updated for updated_action in updated_actions])

        return updated_workflows
