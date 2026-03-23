"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for accessing the readthedocs API (see https://docs.readthedocs.com/platform/stable/api/v3.html).
"""

from os import environ

from core.build_unit import BuildUnit
from util.env import get_env
from util.log import Log
from util.package_manager import PackageManager
from util.requirements import RequirementsFiles


class ReadTheDocsApi:
    """
    Allows to access the readthedocs API.
    """

    ENV_TOKEN = 'RDT_TOKEN'

    class Project:
        """
        Allows to access the readthedocs API for a single project.
        """

        def __init__(self, project_name: str, token: str | None = None):
            """
            :param project_name:    The name of the project
            :param token:           The token to be used for authentication or None, if no token should be used
            """
            self.project_name = project_name
            self.token = token

        def trigger_build(self, version: str):
            """
            Triggers a build.
            """
            import requests

            url = f'https://readthedocs.org/api/v3/projects/{self.project_name}/versions/{version}/builds/'
            Log.verbose(f'Sending request POST {url}')
            token = self.token
            headers = {'Authorization': f'Token {token}'} if token else None
            response = requests.post(url, headers=headers, timeout=5)

            if response.ok:
                Log.verbose(
                    f'Request succeeded with status code {response.status_code} and response: {response.content}'
                )
            else:
                Log.error(f'Request POST {url} failed with status code {response.status_code}')

    @staticmethod
    def __get_token_from_env() -> str | None:
        token = get_env(environ, ReadTheDocsApi.ENV_TOKEN)

        if not token:
            Log.error(
                f'No readthedocs API token is set. You can specify it via the environment variable '
                f'{ReadTheDocsApi.ENV_TOKEN}.'
            )

        return token

    def __init__(self, build_unit: BuildUnit):
        """
        :param build_unit: The build unit to access the readthedocs API from
        """
        PackageManager.install_packages(RequirementsFiles.for_build_unit(build_unit), 'requests')
        self.token = self.__get_token_from_env()

    def set_project(self, project_name: str) -> Project:
        """
        Specifies the name of the project for which the API should be accessed.

        :param project_name:    The name of the project
        :return:                A `ReadTheDocsApi.Project`
        """
        return ReadTheDocsApi.Project(project_name=project_name, token=self.token)
