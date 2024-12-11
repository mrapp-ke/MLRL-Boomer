"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for accessing the GitHub API via "pygithub".
"""
from typing import Optional

from core.build_unit import BuildUnit
from util.pip import Pip


class GithubApi:
    """
    Allows to access the GitHub API.
    """

    class Repository:
        """
        Allows to query information about a single GitHub repository.
        """

        def __init__(self, repository_name: str, authentication):
            """
            :param repository_name: The name of the repository
            :param authentication:  The authentication to be used for accessing the repository or None, if no
                                    authentication should be used
            """
            self.repository_name = repository_name
            self.authentication = authentication

        def get_latest_release_tag(self) -> Optional[str]:
            """
            Returns the tag of the repository's latest release, if any.

            :return: The tag of the latest release or None, if no release is available
            """
            # pylint: disable=import-outside-toplevel
            from github import Github, UnknownObjectException

            with Github(auth=self.authentication) as client:
                try:
                    repository = client.get_repo(self.repository_name)
                    latest_release = repository.get_latest_release()
                    return latest_release.tag_name
                except UnknownObjectException as error:
                    raise RuntimeError('Failed to query latest release of GitHub repository "' + self.repository_name
                                       + '"') from error

    def __init__(self, build_unit: BuildUnit):
        """
        :param build_unit: The build unit to access the GitHub API from
        """
        Pip.for_build_unit(build_unit).install_packages('pygithub')
        self.authentication = None

    def set_token(self, token: Optional[str]) -> 'GithubApi':
        """
        Sets a token to be used for authentication.

        :param token:   The token to be set or None, if no token should be used
        :return:        The `GithubApi` itself
        """
        # pylint: disable=import-outside-toplevel
        from github import Auth
        self.authentication = Auth.Token(token) if token else None
        return self

    def open_repository(self, repository_name: str) -> Repository:
        """
        Specifies the name of a GitHub repository about which information should be queried.

        :param repository_name: The name of the repository, e.g., "mrapp-ke/MLRL-Boomer"
        :return:                A `GithubApi.Repository`
        """
        return GithubApi.Repository(repository_name, self.authentication)
