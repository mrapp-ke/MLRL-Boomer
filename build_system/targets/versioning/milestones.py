"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for managing GitHub milestones.
"""

from os import environ

from core.build_unit import BuildUnit
from util.env import get_env
from util.log import Log
from util.pygithub import GithubApi
from util.version import Version


def __get_repository_name_from_env() -> str | None:
    env_repository = 'GITHUB_REPOSITORY'
    repository_name = get_env(environ, env_repository)

    if not repository_name:
        Log.error(f'The name of a repository must be specified via the environment variable {env_repository}')

    return repository_name


def __get_milestone_from_env() -> str | None:
    env_milestone = 'MILESTONE'
    milestone = get_env(environ, env_milestone)

    if not milestone:
        Log.error(f'The name of a milestone must be specified via the environment variable {env_milestone}')

    return milestone


def close_milestone(build_unit: BuildUnit):
    """
    Closes all milestone that correspond to a version number equal to or less than a version number specified via the
    environment variable "MILESTONE".

    :param build_unit: The build unit this function is called from
    """
    repository_name = __get_repository_name_from_env()
    milestone = __get_milestone_from_env()

    if repository_name and milestone:
        milestone_version = Version.parse(milestone)
        Log.info(
            f'Closing milestones of repository "{repository_name}" corresponding to version "{milestone_version}" or '
            f'smaller...'
        )
        github_api = GithubApi(build_unit).set_token_from_env()
        repository = github_api.open_repository(repository_name)

        for milestone in repository.get_all_milestones():
            milestone_title = milestone.title

            try:
                if Version.parse(milestone_title) <= milestone_version:
                    Log.info(f'Closing milestone "{milestone_title}"...')
                    milestone.edit(title=milestone_title, state='closed')
            except ValueError:
                Log.verbose(f'Ignoring milestone "{milestone_title}", as it does not correspond to a version number.')
