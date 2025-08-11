"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing extensions that add functionality to the command line API provided by this software
package.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.modes import Mode
from mlrl.testbed.modes.mode_batch import BatchMode

from mlrl.util.cli import Argument


class Extension(ABC):
    """
    An abstract base class for all extensions that add functionality to the command line API.
    """

    def __init__(self, *dependencies: 'Extension'):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        self._dependencies = set(dependencies)

    def get_dependencies(self, mode: Mode) -> Set['Extension']:
        """
        Returns a set that contains all extensions, this extension depends on recursively, including only those that
        support a given mode of operation.

        :param mode:    The mode to be supported
        :return:        A set that contains all extensions, this extension depends on
        """
        supported_dependencies = {dependency for dependency in self._dependencies if dependency.is_mode_supported(mode)}
        dependencies = set(supported_dependencies)

        for dependency in self._dependencies:
            for nested_dependency in dependency.get_dependencies(mode):
                dependencies.add(nested_dependency)

        return dependencies

    def get_arguments(self, mode: Mode) -> Set[Argument]:
        """
        Returns a set that contains the arguments that should be added to the command line API according to this
        extension, if it supported a given mode of operation. Dependencies that support the given mode are taken into
        account recursively.

        :param mode:    The mode to be supported
        :return:        A set that contains the arguments
        """
        arguments = self._get_arguments() if self.is_mode_supported(mode) else set()

        for dependency in self._dependencies:
            for argument in dependency.get_arguments(mode):
                arguments.add(argument)

        return arguments

    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        May be overridden by subclasses in order to return the modes of operation supported by this extension.

        :return: A set that contains the supported modes or an empty set, if all modes are supported
        """
        return set()

    def is_mode_supported(self, mode: Mode) -> bool:
        """
        Returns whether this extension supports a given mode of operation or not.

        :param mode:    The mode to be checked
        :return:        True, if the extension supports the given mode, False otherwise
        """
        supported_modes = self.get_supported_modes()
        return type(mode) in supported_modes if supported_modes else True

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        May be overridden by subclasses in order to configure an experiment according to the command line arguments
        specified by the user.

        :param args:                The command line arguments specified by the user
        :param experiment_builder:  A builder that allows to configure the experiment
        """

    def configure_batch_mode(self, args: Namespace, batch_mode: BatchMode):
        """
        May be overridden by subclasses in order to configure the batch mode according to the command line arguments
        specified by the user.

        :param args:        The command line arguments specified by the user
        :param batch_mode:  The batch mode to be configured
        """

    @abstractmethod
    def _get_arguments(self) -> Set[Argument]:
        """
        Must be implemented by subclasses in order to return the arguments that should be added to the command line API
        according to this extension.

        :return: A set that contains the arguments that should be added to the command line API
        """

    @override
    def __hash__(self) -> int:
        return hash(type(self))

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))


class NopExtension(Extension):
    """
    An extension that does nothing.
    """

    @override
    def _get_arguments(self) -> Set[Argument]:
        return set()
