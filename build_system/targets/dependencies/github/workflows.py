"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility classes for accessing the project's GitHub workflows.
"""
from functools import cached_property
from typing import Any, Dict, List, Set

from core.build_unit import BuildUnit

from targets.dependencies.github.modules import GithubWorkflowModule
from targets.dependencies.github.pyyaml import YamlFile


class Workflow(YamlFile):
    """
    A GitHub workflow.
    """

    @staticmethod
    def find_tags(yaml_dict: Dict, *tags: str) -> List[Any]:
        """
        Returns a list that contains the values of all tags with a specific name in a given dictionary that stores parts
        of a YAML file.

        :param yaml_dict:   A dictionary that stores the content of the YAML file
        :param tags:        The names of the tags
        :return:            A list that contains the values of all tags that have been found
        """
        tag_names = set(tags)
        values = []

        for key, value in yaml_dict.items():
            if key in tag_names:
                values.append(value)
            elif isinstance(value, dict):
                values.extend(Workflow.find_tags(value, *tag_names))

        return values

    @staticmethod
    def find_tag(yaml_dict: Dict, tag: str, default: Any = None) -> Any:
        """
        Returns the value of the first tag with a specific name in a given dictionary that stores parts of a YAML file.

        :param yaml_dict:   A dictionary that stores the content of the YAML file
        :param tag:         The name of the tag
        :param default:     An optional default value to be returned if no tag with the given name is found
        :return:            The value of the tag that has been found or the given default value
        """
        values = Workflow.find_tags(yaml_dict, tag)
        return values[0] if values else default


class Workflows:
    """
    Allows to access the workflows in a `GithubWorkflowModule`.
    """

    def __init__(self, build_unit: BuildUnit, module: GithubWorkflowModule):
        """
        :param build_unit:  The build unit from which workflow definition files should be read
        :param module:      The module, that contains the workflow definition files
        """
        self.build_unit = build_unit
        self.module = module

    @cached_property
    def workflows(self) -> Set[Workflow]:
        """
        All GitHub workflows that are defined in the directory where workflow definition files are located.
        """
        return {Workflow(self.build_unit, workflow_file) for workflow_file in self.module.find_workflow_files()}
