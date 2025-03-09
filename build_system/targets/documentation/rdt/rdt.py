"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for accessing the readthedocs API (see https://docs.readthedocs.com/platform/stable/api/v3.html).
"""
from os import environ
from typing import Optional

from core.build_unit import BuildUnit
from util.env import get_env
from util.log import Log
from util.pip import Pip


class ReadTheDocsApi:
    """
    Allows to access the readthedocs API.
    """

    ENV_TOKEN = 'RDT_TOKEN'

    class Project:
        """
        Allows to access the readthedocs API for a single project.
        """

        def __init__(self, project_name: str, token: str):
            """
            :param project_name:    The name of the project
            :param token:           The token to be used for authentication
            """
            self.project_name = project_name
            self.token = token

        def trigger_build(self, version: str):
            """
            Triggers a build.
            """
            # pylint: disable=import-outside-toplevel
            import requests
            url = 'https://readthedocs.org/api/v3/projects/' + self.project_name + '/versions/' + version + '/builds/'
            Log.verbose('Sending request POST %s', url)
            response = requests.post(url, headers={'Authorization': 'Token ' + self.token}, timeout=5)

            if response.ok:
                Log.verbose('Request succeeded with status code %s and response: %s', response.status_code,
                            response.content)
            else:
                Log.error('Request failed with status code %s', response.status_code)

    @staticmethod
    def __get_token_from_env() -> Optional[str]:
        token = get_env(environ, ReadTheDocsApi.ENV_TOKEN)

        if not token:
            Log.error('No readthedocs API token is set. You can specify it via the environment variable %s.',
                      ReadTheDocsApi.ENV_TOKEN)

        return token

    def __init__(self, build_unit: BuildUnit):
        """
        :param build_unit: The build unit to access the readthedocs API from
        """
        Pip.for_build_unit(build_unit).install_packages('requests')
        self.token = self.__get_token_from_env()

    def set_project(self, project_name: str) -> Project:
        """
        Specifies the name of the project for which the API should be accessed.

        :param project_name:    The name of the project
        :return:                A `ReadTheDocsApi.Project`
        """
        return ReadTheDocsApi.Project(project_name=project_name, token=self.token)
