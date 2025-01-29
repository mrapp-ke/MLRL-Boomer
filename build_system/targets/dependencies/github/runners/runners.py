"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking the project's GitHub workflows for outdated runner images (see
https://docs.github.com/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners).
"""
import re

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree

from core.build_unit import BuildUnit
from util.log import Log
from util.pip import Pip

from targets.dependencies.github.modules import GithubWorkflowModule
from targets.dependencies.github.workflows import Workflow, Workflows


@dataclass
class RunnerVersion:
    """
    The version of a GitHub-hosted runner.

    Attributes:
        version: The full version string
    """
    version: str

    SEPARATOR = '.'

    def is_latest(self) -> bool:
        """
        Returns whether the version is the latest version of a runner, False otherwise.

        :return: True, if the version is the latest, False otherwise
        """
        return self.version == 'latest'

    @property
    def version_numbers(self) -> List[int]:
        """
        A list that stores the individual version numbers, the full version consists of.
        """
        return [] if self.is_latest() else [int(version_number) for version_number in str(self).split(self.SEPARATOR)]

    def __str__(self) -> str:
        return self.version

    def __lt__(self, other: 'RunnerVersion') -> bool:
        if not self.is_latest():
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
class Runner:
    """
    A GitHub-hosted runner.

    Attributes:
        image:          The name of the image used by the runner
        version:        The version of the runner
        architecture:   The architecture of the image or None, if the architecture is unknown
    """
    image: str
    version: RunnerVersion
    architecture: Optional[str] = None

    SEPARATOR = '-'

    @staticmethod
    def parse(text: str) -> 'Runner':
        """
        Creates and returns a GitHub-hosted runner from a given text.

        :param text:    The text to be parsed
        :return:        The GitHub-hosted runner that has been created
        """
        parts = text.split(Runner.SEPARATOR)

        if len(parts) < 2 or len(parts) > 3:
            raise ValueError('Runner must contain the symbol + "' + Runner.SEPARATOR + '" once or twice, but got "'
                             + text + '"')

        return Runner(image=parts[0],
                      version=RunnerVersion(parts[1]),
                      architecture=parts[2] if len(parts) > 2 else None)

    @property
    def name(self) -> str:
        """
        The name of the runner without the version.
        """
        return self.image + (self.SEPARATOR + self.architecture if self.architecture else '')

    def __str__(self) -> str:
        result = self.image + self.SEPARATOR + str(self.version)

        if self.architecture:
            result += self.SEPARATOR + self.architecture

        return result

    def __eq__(self, other: 'Runner') -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class Runners(Workflow):
    """
    Allows to access and update GitHub-hosted runners used in a workflow.
    """

    def __parse_runs_on_clause(self, runs_on_clause: str) -> Runner:
        try:
            return Runner.parse(runs_on_clause)
        except ValueError as error:
            raise RuntimeError('Failed to parse runs-on-clause in workflow "' + self.file + '"') from error

    def __parse_strategy(self, strategy: Dict) -> Set[Runner]:
        runners = set()

        for os in self.find_tag(strategy, 'os', default=[]):
            try:
                runners.add(Runner.parse(os))
            except ValueError as error:
                raise RuntimeError('Failed to parse strategy.matrix.os-clause in workflow "' + self.file
                                   + '"') from error

        return runners

    @cached_property
    def runners(self) -> Set[Runner]:
        """
        A set that contains all GitHub-hosted runners used in the workflow.
        """
        runners = set()

        for job in self.find_tag(self.yaml_dict, 'jobs', default={}).values():
            runs_on_clause = self.find_tag(job, 'runs-on')

            if runs_on_clause:
                if runs_on_clause.replace(' ', '') == '${{matrix.os}}':
                    strategy = self.find_tag(job, 'strategy')

                    if strategy:
                        runners.update(self.__parse_strategy(strategy))
                else:
                    runners.add(self.__parse_runs_on_clause(runs_on_clause))

        return runners

    def update_runners(self, *updated_runners: Runner):
        """
        Updates given runners in the workflow definition file.

        :param updated_runners: The runners to be updated
        """
        updated_lines = []

        for line in self.lines:
            for updated_runner in updated_runners:
                architecture = updated_runner.architecture
                regex = updated_runner.image + '-[0-9]+(.[0-9]+)*' + ('-' + architecture if architecture else '')
                line = re.sub(regex, str(updated_runner), line)

            updated_lines.append(line)

        self.write_lines(*updated_lines)

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.runners
        except AttributeError:
            pass


class RunnerUpdater(Workflows):
    """
    Allows checking the versions of GitHub-hosted runners used in multiple workflows and updating outdated ones.
    """

    @dataclass
    class OutdatedRunner:
        """
        An outdated GitHub-hosted runner.

        Attributes:
            runner:         The outdated runner
            latest_version: The latest version of the runner
        """
        runner: Runner
        latest_version: RunnerVersion

        def __str__(self) -> str:
            return str(self.runner)

        def __eq__(self, other: 'RunnerUpdater.OutdatedRunner') -> bool:
            return self.runner == other.runner

        def __hash__(self) -> int:
            return hash(self.runner)

    @dataclass
    class UpdatedRunner:
        """
        A GitHub-hosted runner that has been updated.

        Attributes:
            previous:   The previous runner
            updated:    The updated runner
        """
        previous: 'RunnerUpdater.OutdatedAction'
        updated: Runner

        def __str__(self) -> str:
            return str(self.updated)

        def __eq__(self, other: 'RunnerUpdater.UpdatedRunner') -> bool:
            return self.updated == other.updated

        def __hash__(self) -> int:
            return hash(self.updated)

    def __download_runner_documentation(self) -> str:
        Pip.for_build_unit(self.build_unit).install_packages('requests')
        # pylint: disable=import-outside-toplevel
        import requests
        repository = 'github/docs'
        file = 'data/reusables/actions/supported-github-runners.md'
        url = 'https://raw.githubusercontent.com/' + repository + '/refs/heads/main/' + file
        response = requests.get(url, timeout=5)

        if not response.ok:
            Log.error('Failed to download list of runners from "%s". Received status code "%s".', url,
                      response.status_code)

        return response.text

    @staticmethod
    def __find_relevant_section(lines: List[str]) -> List[str]:
        i = 0

        for i, line in enumerate(lines):
            if line.startswith('#') and line.find('runners for public repositories') > 0:
                break

        relevant_lines = []

        for line in lines[(i + 1):]:
            if line.startswith('#'):
                break

            relevant_lines.append(line)

        return relevant_lines

    @staticmethod
    def __find_table(lines: List[str]) -> List[str]:
        i = 0

        for i, line in enumerate(lines):
            if line.startswith('<table'):
                break

        relevant_lines = []

        for line in lines[i:]:
            relevant_lines.append(line)

            if line.startswith('</table'):
                break

        return relevant_lines

    @staticmethod
    def __parse_table(lines: List[str]) -> Set[Runner]:
        html = '\n'.join(lines)
        table = ElementTree.fromstring(html)
        header = table.find('./thead')
        relevant_column_text = 'workflow label'
        relevant_column_index = None

        for i, column in enumerate(header.findall('.//th')):
            text = column.text if column.text else column.find('.//').text

            if text.lower().find(relevant_column_text) >= 0:
                relevant_column_index = i
                break

        if not relevant_column_index:
            Log.error('Could not find table column with text "%s": %s', relevant_column_text, html)

        runners = set()

        for row in table.findall('./tbody/tr'):
            relevant_column = row.find('./td[' + str(relevant_column_index + 1) + ']')

            for link in relevant_column.findall('.//a[@href]'):
                runners.add(Runner.parse(link.text))

        return runners

    def __get_latest_runners_from_documentation(self) -> Dict[Tuple[str, str], RunnerVersion]:
        Log.info('Retrieving the latest runners from the GitHub documentation...')
        runner_documentation = self.__download_runner_documentation()
        lines = runner_documentation.split('\n')
        lines = self.__find_relevant_section(lines)
        lines = self.__find_table(lines)
        versioned_runners = {runner for runner in self.__parse_table(lines) if not runner.version.is_latest()}
        latest_runners = {}

        for runner in versioned_runners:
            key = (runner.image, runner.architecture)
            version = latest_runners.get(key)

            if not version or version < runner.version:
                latest_runners[key] = runner.version

        if not latest_runners:
            Log.error('Failed to retrieve latest runners from the GitHub documentation!')

        return latest_runners

    def __get_latest_runner_version(self, runner: Runner) -> RunnerVersion:
        if self.version_cache is None:
            self.version_cache = self.__get_latest_runners_from_documentation()

        latest_version = self.version_cache.get((runner.image, runner.architecture))

        if not latest_version:
            Log.error('Latest version of runner "%s" is unknown!', runner)

        return latest_version

    def __init__(self, build_unit: BuildUnit, module: GithubWorkflowModule):
        """
        :param build_unit:  The build unit from which workflow definition files should be read
        :param module:      The module, that contains the workflow definition files
        """
        super().__init__(build_unit, module)
        self.version_cache = None

    def find_outdated_workflows(self) -> Dict[Runners, Set[OutdatedRunner]]:
        """
        Finds and returns all workflows with outdated runners.

        :return: A dictionary that contains for each workflow a set of outdated runners
        """
        outdated_workflows = {}

        for workflow in self.workflows:
            Log.info('Searching for GitHub-hosted runners in workflow "%s"...', workflow.file)
            workflow = Runners(self.build_unit, file=workflow.file)

            for runner in workflow.runners:
                if not runner.version.is_latest():
                    latest_version = self.__get_latest_runner_version(runner)

                    if runner.version < latest_version:
                        outdated_runners = outdated_workflows.setdefault(workflow, set())
                        outdated_runners.add(RunnerUpdater.OutdatedRunner(runner, latest_version))

        return outdated_workflows

    def update_outdated_workflows(self) -> Dict[Runners, Set[UpdatedRunner]]:
        """
        Updates all workflows with outdated GitHub runners.

        :return: A dictionary that contains for each workflow a set of updated runners
        """
        updated_workflows = {}

        for workflow, outdated_runners in self.find_outdated_workflows().items():
            updated_runners = set()

            for outdated_runner in outdated_runners:
                updated_runners = updated_workflows.setdefault(workflow, updated_runners)
                updated_runner = replace(outdated_runner.runner, version=outdated_runner.latest_version)
                updated_runners.add(RunnerUpdater.UpdatedRunner(previous=outdated_runner, updated=updated_runner))

            workflow.update_runners(*[updated_runner.updated for updated_runner in updated_runners])

        return updated_workflows
