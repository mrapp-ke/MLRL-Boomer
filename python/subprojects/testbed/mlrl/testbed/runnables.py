"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from functools import reduce
from typing import List, Optional, Set, override

from mlrl.testbed.arguments import PredictionDatasetArguments
from mlrl.testbed.command import Command
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.output.meta_data.extension import MetaDataExtension
from mlrl.testbed.experiments.problem_domain import ProblemDomain
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode
from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.cli import Argument, CommandLineInterface

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

        @override
        def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
            """
            return {
                PredictionDatasetArguments.PREDICT_FOR_TRAINING_DATA,
                PredictionDatasetArguments.PREDICT_FOR_TEST_DATA,
            }

        @override
        def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: ExperimentMode):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
            """
            experiment_builder.set_predict_for_training_dataset(
                PredictionDatasetArguments.PREDICT_FOR_TRAINING_DATA.get_value(args))
            experiment_builder.set_predict_for_test_dataset(
                PredictionDatasetArguments.PREDICT_FOR_TEST_DATA.get_value(args))

        @override
        def get_supported_modes(self) -> Set[ExperimentMode]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
            """
            return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.RUN}

    def get_extensions(self) -> List[Extension]:
        """
        May be overridden by subclasses in order to return the extensions that should be applied to the runnable.

        :return: A set that contains the extensions to be applied to the runnable
        """
        return [
            Runnable.PredictionDatasetExtension(),
            MetaDataExtension(),
            SlurmExtension(),
        ]

    def get_supported_extensions(self, mode: ExperimentMode) -> List[Extension]:
        """
        Returns the extensions that should be applied to the runnable and support a given mode of operation.

        :param mode:    The mode to be supported
        :return:        A list that contains all extensions that should be applied to the runnable and support the given
                        mode
        """
        return [extension for extension in self.get_extensions() if extension.is_mode_supported(mode)]

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: A `ProgramInfo` or None, if no information is provided
        """
        return None

    def run(self, mode: Mode, arguments: List[Argument], args: Namespace):
        """
        Executes the runnable.

        :param mode:        The mode of operation
        :param arguments:   A list that contains the command line arguments available in the current mode of operation
        :param args:        The command line arguments specified by the user
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
            def create_dataset_splitter(self, args: Namespace, load_dataset: bool = True) -> DatasetSplitter:
                return self.runnable.create_dataset_splitter(args, load_dataset)

            @override
            def create_experiment_builder(self,
                                          experiment_mode: ExperimentMode,
                                          args: Namespace,
                                          command: Command,
                                          load_dataset: bool = True) -> Experiment.Builder:
                runnable = self.runnable
                experiment_builder = runnable.create_experiment_builder(experiment_mode=experiment_mode,
                                                                        args=args,
                                                                        command=command,
                                                                        load_dataset=load_dataset)

                for extension in runnable.get_supported_extensions(experiment_mode):
                    extension.configure_experiment(args, experiment_builder, experiment_mode)

                    for dependency in extension.get_dependencies(experiment_mode):
                        dependency.configure_experiment(args, experiment_builder, experiment_mode)

                return experiment_builder

        mode.run_experiment(arguments, args, RecipeWrapper(self))

    def configure_batch_mode(self, cli: CommandLineInterface) -> BatchMode:
        """
        Configures the batch mode according to the extensions applied to the runnable.

        :param cli: The command line interface to be configured
        """
        batch_mode = BatchMode(self.create_batch_config_file_factory())
        args = cli.parse_known_args()

        for extension in self.get_supported_extensions(batch_mode.to_enum()):
            extension.configure_batch_mode(args, batch_mode)

            for dependency in extension.get_dependencies(batch_mode.to_enum()):
                dependency.configure_batch_mode(args, batch_mode)

        return batch_mode

    def configure_read_mode(self, cli: CommandLineInterface) -> ReadMode:
        """
        Configures the read mode according to the extensions applied to the runnable.

        :param cli: The command line interface to be configured
        """
        read_mode = ReadMode()
        args = cli.parse_known_args()

        for extension in self.get_supported_extensions(read_mode.to_enum()):
            extension.configure_read_mode(args, read_mode)

            for dependency in extension.get_dependencies(read_mode.to_enum()):
                dependency.configure_read_mode(args, read_mode)

        return read_mode

    def configure_run_mode(self, cli: CommandLineInterface) -> RunMode:
        """
        Configures the run mode according to the extensions applied to the runnable.

        :param cli: The command line interface to be configured
        """
        run_mode = RunMode()
        args = cli.parse_known_args()

        for extension in self.get_supported_extensions(run_mode.to_enum()):
            extension.configure_run_mode(args, run_mode)

            for dependency in extension.get_dependencies(run_mode.to_enum()):
                dependency.configure_run_mode(args, run_mode)

        return run_mode

    def configure_arguments(self, cli: CommandLineInterface, mode: Mode):
        """
        Configures the command line interface according to the extensions applied to the runnable.

        :param cli:     The command line interface to be configured
        :param mode:    The mode of operation
        """
        extension_arguments: Set[Argument] = reduce(
            lambda aggr, extension: aggr | extension.get_arguments(mode.to_enum()), self.get_extensions(), set())
        algorithmic_arguments = self.get_algorithmic_arguments(cli.parse_known_args())
        mode.configure_arguments(cli,
                                 extension_arguments=sorted(extension_arguments, key=lambda arg: arg.name),
                                 algorithmic_arguments=sorted(algorithmic_arguments, key=lambda arg: arg.name))

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
