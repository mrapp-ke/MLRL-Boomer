"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing extensions that add functionality to the command line API provided by this software
package.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from functools import cached_property
from typing import List

from mlrl.testbed.experiments.experiment import Experiment

from mlrl.util.cli import Argument


class Extension(ABC):
    """
    An abstract base class for all extensions that add functionality to the command line API.
    """

    def __init__(self, *dependencies: 'Extension'):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        self._dependencies = list(dependencies)

    @cached_property
    def dependencies(self) -> List['Extension']:
        """
        A list that contains all extensions, this extension depends on recursively.
        """
        dependencies = {}

        for dependency in self._dependencies:
            dependencies.setdefault(type(dependency), dependency)

            for nested_dependency in dependency.dependencies:
                dependencies.setdefault(type(nested_dependency).__name__, nested_dependency)

        return list(dependencies.values())

    @cached_property
    def arguments(self) -> List[Argument]:
        """
        A list that contains the arguments that should be added to the command line API according to this extension,
        also taking into account dependencies recursively.
        """
        arguments = {argument.key: argument for argument in self._get_arguments()}

        for dependency in self._dependencies:
            for argument in dependency.arguments:
                arguments.setdefault(argument.key, argument)

        return list(arguments.values())

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        May be overridden by subclasses in order to configure an experiment according to the command line arguments
        specified by the user.

        :param args:                The command line arguments specified by the user
        :param experiment_builder:  A builder that allows to configure the experiment
        """

    @abstractmethod
    def _get_arguments(self) -> List[Argument]:
        """
        Must be implemented by subclasses in order to return the arguments that should be added to the command line API
        according to this extension.

        :return: A list that contains the arguments that should be added to the command line API
        """
