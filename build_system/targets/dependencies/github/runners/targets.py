"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for updating the project's GitHub runners.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from util.log import Log

from targets.dependencies.github.modules import GithubWorkflowModule
from targets.dependencies.github.runners.runners import RunnerUpdater
from targets.dependencies.table import Table

MODULE_FILTER = GithubWorkflowModule.Filter()


class CheckGithubRunners(PhonyTarget.Runnable):
    """
    Prints all outdated runners used in the project's GitHub workflows.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        outdated_workflows = RunnerUpdater(build_unit, module).find_outdated_workflows()

        if outdated_workflows:
            table = Table(build_unit, 'Workflow', 'Runner', 'Current version', 'Latest version')

            for workflow, outdated_runners in outdated_workflows.items():
                for outdated_runner in outdated_runners:
                    table.add_row(workflow.file, str(outdated_runner.runner.name), str(outdated_runner.runner.version),
                                  str(outdated_runner.latest_version))

            table.sort_rows(0, 1)
            Log.info('The following GitHub-hosted runners are outdated:\n\n%s', str(table))
        else:
            Log.info('All GitHub-hosted runners are up-to-date!')


class UpdateGithubRunners(PhonyTarget.Runnable):
    """
    Updates and prints all outdated runners used in the project's GitHub workflows.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        updated_workflows = RunnerUpdater(build_unit, module).update_outdated_workflows()

        if updated_workflows:
            table = Table(build_unit, 'Workflow', 'Runner', 'Previous version', 'Updated version')

            for workflow, updated_runners in updated_workflows.items():
                for updated_runner in updated_runners:
                    table.add_row(workflow.file, updated_runner.updated.name,
                                  str(updated_runner.previous.runner.version), str(updated_runner.updated.version))

            table.sort_rows(0, 1)
            Log.info('The following GitHub-hosted runners have been updated:\n\n%s', str(table))
        else:
            Log.info('No GitHub-hosted runners have been updated.')
