"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to GitHub workflows tha belong to individual modules.
"""
from typing import List

from core.modules import Module
from util.files import FileSearch, FileType


class GithubWorkflowModule(Module):
    """
    A module that contains GitHub workflows.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules that contain GitHub workflows.
        """

        def matches(self, module: Module) -> bool:
            return isinstance(module, GithubWorkflowModule)

    def __init__(self, root_directory: str, workflow_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param workflow_file_search:    The `FileSearch` that should be used to search for workflow definition files
        """
        self.root_directory = root_directory
        self.workflow_file_search = workflow_file_search

    def find_workflow_files(self) -> List[str]:
        """
        Finds and returns all workflow definition files that belong to the module.

        :return: A list that contains the paths of the workflow definition files that have been found
        """
        return self.workflow_file_search.filter_by_file_type(FileType.yaml()).list(self.root_directory)

    def __str__(self) -> str:
        return 'GithubWorkflowModule {root_directory="' + self.root_directory + '"}'
