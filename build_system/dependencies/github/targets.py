"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for updating the project's GitHub Actions.
"""
from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import PhonyTarget
from dependencies.github.actions import WorkflowUpdater
from dependencies.github.modules import GithubWorkflowModule
from dependencies.table import Table
from util.log import Log

MODULE_FILTER = GithubWorkflowModule.Filter()


class CheckGithubActions(PhonyTarget.Runnable):
    """
    Prints all outdated Actions used in the project's GitHub workflows.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        outdated_workflows = WorkflowUpdater(build_unit, module).find_outdated_workflows()

        if outdated_workflows:
            table = Table(build_unit, 'Workflow', 'Action', 'Current version', 'Latest version')

            for workflow, outdated_actions in outdated_workflows.items():
                for outdated_action in outdated_actions:
                    table.add_row(workflow.file, str(outdated_action.action.name), str(outdated_action.action.version),
                                  str(outdated_action.latest_version))

            table.sort_rows(0, 1)
            Log.info('The following GitHub Actions are outdated:\n\n%s', str(table))
        else:
            Log.info('All GitHub Actions are up-to-date!')


class UpdateGithubActions(PhonyTarget.Runnable):
    """
    Updates and prints all outdated Actions used in the project's GitHub workflows.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        updated_workflows = WorkflowUpdater(build_unit, module).update_outdated_workflows()

        if updated_workflows:
            table = Table(build_unit, 'Workflow', 'Action', 'Previous version', 'Updated version')

            for workflow, updated_actions in updated_workflows.items():
                for updated_action in updated_actions:
                    table.add_row(workflow.file, updated_action.updated.name,
                                  str(updated_action.previous.action.version), str(updated_action.updated.version))

            table.sort_rows(0, 1)
            Log.info('The following GitHub Actions have been updated:\n\n%s', str(table))
        else:
            Log.info('No GitHub Actions have been updated.')
