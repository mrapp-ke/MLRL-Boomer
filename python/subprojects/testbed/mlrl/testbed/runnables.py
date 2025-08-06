"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from functools import cached_property, reduce
from typing import List, Optional, Set, override

from mlrl.testbed.command import Command
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.output.meta_data.extension import MetaDataExtension
from mlrl.testbed.experiments.problem_domain import ProblemDomain
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.extensions import Extension
from mlrl.testbed.extensions.extension_log import LogExtension
from mlrl.testbed.modes import Mode
from mlrl.testbed.modes.mode_batch import BatchMode
from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.cli import Argument, BoolArgument, CommandLineInterface

try:
    from mlrl.testbed_slurm.extension import SlurmExtension
except ImportError:
    # Dependency 'mlrl-testbed' is not available, but that's okay since it's optional.
    from mlrl.testbed.extensions import NopExtension as SlurmExtension


class Runnable(Recipe, ABC):
    """
    An abstract base class for all programs that can be configured via the command line API. The programs functionality
    is implemented by individual extensions that are applied to the runnable.
    """

    class PredictionDatasetExtension(Extension):
        """
        An extension that configures the functionality to predict for different datasets.
        """

        PREDICT_FOR_TRAINING_DATA = BoolArgument(
            '--predict-for-training-data',
            default=False,
            description='Whether predictions should be obtained for the training data or not.',
        )

        PREDICT_FOR_TEST_DATA = BoolArgument(
            '--predict-for-test-data',
            default=True,
            description='Whether predictions should be obtained for the test data or not.',
        )

        @override
        def _get_arguments(self) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {self.PREDICT_FOR_TRAINING_DATA, self.PREDICT_FOR_TEST_DATA}

        @override
        def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
            """
            experiment_builder.set_predict_for_training_dataset(self.PREDICT_FOR_TRAINING_DATA.get_value(args))
            experiment_builder.set_predict_for_test_dataset(self.PREDICT_FOR_TEST_DATA.get_value(args))

    @cached_property
    def extensions(self) -> List[Extension]:
        """
        A list that contains the extensions that should be applied to the runnable sorted in a consistent order.
        """
        return sorted(self.get_extensions(), key=lambda extension: type(extension).__name__)

    def get_extensions(self) -> Set[Extension]:
        """
        May be overridden by subclasses in order to return the extensions that should be applied to the runnable.

        :return: A set that contains the extensions to be applied to the runnable
        """
        return {
            Runnable.PredictionDatasetExtension(),
            LogExtension(),
            MetaDataExtension(),
            SlurmExtension(),
        }

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: A `ProgramInfo` or None, if no information is provided
        """
        return None

    def run(self, mode: Mode, args: Namespace):
        """
        Executes the runnable.

        :param mode:    The mode of operation
        :param args:    The command line arguments specified by the user
        """

        class RecipeWrapper(Recipe):
            """
            A `Recipe` that wraps a `Runnable`.
            """

            def __init__(self, runnable: Runnable):
                """
                :param runnable: The `Runnable` to be wrapped
                """
                self.runnable = runnable

            @override
            def create_problem_domain(self, args: Namespace) -> ProblemDomain:
                return self.runnable.create_problem_domain(args)

            @override
            def create_dataset_splitter(self, args: Namespace) -> DatasetSplitter:
                return self.runnable.create_dataset_splitter(args)

            @override
            def create_experiment_builder(self, args: Namespace, command: Command) -> Experiment.Builder:
                runnable = self.runnable
                experiment_builder = runnable.create_experiment_builder(args, command)

                for extension in runnable.extensions:
                    extension.configure_experiment(args, experiment_builder)

                    for dependency in extension.get_dependencies(mode):
                        dependency.configure_experiment(args, experiment_builder)

                return experiment_builder

        mode.run_experiment(args, RecipeWrapper(self))

    def configure_batch_mode(self, cli: CommandLineInterface) -> BatchMode:
        """
        Configures the batch mode according to the extensions applied to the runnable.
        """
        batch_mode = BatchMode(self.create_batch_config_file_factory())
        args = cli.parse_known_args()

        for extension in self.extensions:
            extension.configure_batch_mode(args, batch_mode)

            for dependency in extension.get_dependencies(batch_mode):
                dependency.configure_batch_mode(args, batch_mode)

        return batch_mode

    def configure_arguments(self, cli: CommandLineInterface, mode: Mode):
        """
        Configures the command line interface according to the extensions applied to the runnable.

        :param cli:     The command line interface to be configured
        :param mode:    The mode of operation
        """
        arguments = reduce(lambda aggr, extension: aggr | extension.get_arguments(mode), self.extensions, set())
        arguments.update(self.get_algorithmic_arguments(cli.parse_known_args()))
        cli.add_arguments(*sorted(arguments, key=lambda arg: arg.name))

    # pylint: disable=unused-argument
    def get_algorithmic_arguments(self, known_args: Namespace) -> Set[Argument]:
        """
        May be overridden by subclasses in order to return the arguments for configuring algorithmic parameters that
        should be added to the command line API.

        :param known_args:  The command line arguments specified by the user
        :return:            A set that contains the arguments that should be added to the command line API
        """
        return set()

    @abstractmethod
    def create_batch_config_file_factory(self) -> BatchMode.ConfigFile.Factory:
        """
        Must be implemented by subclasses in order to create the factory that allows to create the configuration file
        that configures the batch of experiments to be run in batch mode.

        :return: The factory that has been created
        """
