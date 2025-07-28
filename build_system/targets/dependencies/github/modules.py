"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to GitHub workflows.
"""
from pathlib import Path
from typing import List, override

from core.modules import Module, ModuleRegistry
from util.files import FileSearch, FileType


class GithubWorkflowModule(Module):
    """
    A module that provides access to GitHub workflows.
    """

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `GithubWorkflowModule`.
        """

        @override
        def matches(self, module: Module, _: ModuleRegistry) -> bool:
            return isinstance(module, GithubWorkflowModule)

    def __init__(self, root_directory: Path, workflow_file_search: FileSearch = FileSearch().set_recursive(True)):
        """
        :param root_directory:          The path to the module's root directory
        :param workflow_file_search:    The `FileSearch` that should be used to search for workflow definition files
        """
        self.root_directory = root_directory
        self.workflow_file_search = workflow_file_search

    def find_workflow_files(self) -> List[Path]:
        """
        Finds and returns all workflow definition files that belong to the module.

        :return: A list that contains the paths of the workflow definition files that have been found
        """
        return self.workflow_file_search.filter_by_file_type(FileType.yaml()).list(self.root_directory)

    @override
    def __str__(self) -> str:
        return 'GithubWorkflowModule {root_directory="' + str(self.root_directory) + '"}'
