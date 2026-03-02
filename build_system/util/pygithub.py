"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for accessing the GitHub API via "pygithub".
"""
from os import environ
from typing import Any, Iterable, Optional

from core.build_unit import BuildUnit
from util.env import get_env
from util.log import Log
from util.package_manager import PackageManager
from util.requirements import RequirementsFiles


class GithubApi:
    """
    Allows to access the GitHub API.
    """

    ENV_GITHUB_TOKEN = 'GITHUB_TOKEN'

    class Repository:
        """
        Allows to query information about a single GitHub repository.
        """

        def __create_client(self) -> Any:
            # pylint: disable=import-outside-toplevel
            from github import Auth, Github
            return Github(auth=Auth.Token(self.token) if self.token else None)

        def __init__(self, repository_name: str, token: Optional[str] = None):
            """
            :param repository_name: The name of the repository
            :param token:           The API token to be used for accessing the repository or None, if no token should be
                                    used
            """
            self.repository_name = repository_name
            self.token = token

        def get_latest_release(self) -> Optional[Any]:
            """
            Returns repository's latest release, if any.

            :return: The latest release or None, if no release is available
            """

            with self.__create_client() as client:
                try:
                    repository = client.get_repo(self.repository_name)
                    return repository.get_latest_release()
                except Exception as error:
                    raise RuntimeError('Failed to query latest release of GitHub repository "' + self.repository_name
                                       + '"') from error

        def get_all_releases(self) -> Iterable[Any]:
            """
            Returns all releases of the repository.

            :return: The releases
            """
            with self.__create_client() as client:
                try:
                    repository = client.get_repo(self.repository_name)
                    return repository.get_releases()
                except Exception as error:
                    raise RuntimeError('Failed to query releases of GitHub repository "' + self.repository_name
                                       + '"') from error

        def get_all_milestones(self, state='open') -> Iterable[Any]:
            """
            Returns all milestones of the repository.

            :param state:   The state of the milestones to be returned. Must be 'open', 'closed', or 'all'
            :return:        The milestones
            """
            with self.__create_client() as client:
                try:
                    repository = client.get_repo(self.repository_name)
                    return repository.get_milestones(state=state)
                except Exception as error:
                    raise RuntimeError('Failed to query milestones of GitHub repository "'
                                       + self.repository_name) from error

    def __init__(self, build_unit: BuildUnit):
        """
        :param build_unit: The build unit to access the GitHub API from
        """
        PackageManager.install_packages(RequirementsFiles.for_build_unit(build_unit), 'pygithub')
        self.token: Optional[str] = None

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
