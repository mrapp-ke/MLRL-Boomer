"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for accessing the GitHub API via "pygithub".
"""
from os import environ
from typing import Any, Optional

from core.build_unit import BuildUnit
from util.env import get_env
from util.log import Log
from util.pip import Pip


class GithubApi:
    """
    Allows to access the GitHub API.
    """

    ENV_GITHUB_TOKEN = 'GITHUB_TOKEN'

    class Repository:
        """
        Allows to query information about a single GitHub repository.
        """

        def __init__(self, repository_name: str, token: str):
            """
            :param repository_name: The name of the repository
            :param token:           The API token to be used for accessing the repository or None, if no token should be
                                    used
            """
            self.repository_name = repository_name
            self.token = token

        def get_latest_release(self) -> Optional[Any]:
            """
            Returns the tag of the repository's latest release, if any.

            :return: The tag of the latest release or None, if no release is available
            """
            # pylint: disable=import-outside-toplevel
            from github import Auth, Github, UnknownObjectException

            with Github(auth=Auth.Token(self.token) if self.token else None) as client:
                try:
                    repository = client.get_repo(self.repository_name)
                    return repository.get_latest_release()
                except UnknownObjectException as error:
                    raise RuntimeError('Failed to query latest release of GitHub repository "' + self.repository_name
                                       + '"') from error

    def __init__(self, build_unit: BuildUnit):
        """
        :param build_unit: The build unit to access the GitHub API from
        """
        Pip.for_build_unit(build_unit).install_packages('pygithub')
        self.token = None

    def set_token(self, token: Optional[str]) -> 'GithubApi':
        """
        Sets a token to be used for authentication.

        :param token:   The token to be set or None, if no token should be used
        :return:        The `GithubApi` itself
        """
        self.token = token
        return self

    def set_token_from_env(self) -> 'GithubApi':
        """
        Obtains and sets a token to be used for authentication from the environment variable `ENV_GITHUB_TOKEN`.

        :return: The `GithubApi` itself
        """
        github_token = get_env(environ, self.ENV_GITHUB_TOKEN)

        if not github_token:
            Log.warning('No GitHub API token is set. You can specify it via the environment variable %s.',
                        self.ENV_GITHUB_TOKEN)

        return self.set_token(github_token)

    def open_repository(self, repository_name: str) -> Repository:
        """
        Specifies the name of a GitHub repository about which information should be queried.

        :param repository_name: The name of the repository, e.g., "mrapp-ke/MLRL-Boomer"
        :return:                A `GithubApi.Repository`
        """
        return GithubApi.Repository(repository_name, self.token)
