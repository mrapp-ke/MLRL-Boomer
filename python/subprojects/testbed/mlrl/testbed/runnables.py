"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List, Optional

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.extensions import Extension
from mlrl.testbed.extensions.extension_log import LogExtension
from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.cli import Argument, BoolArgument, CommandLineInterface


class Runnable(ABC):
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

        def get_arguments(self) -> List[Argument]:
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
            """
            return [self.PREDICT_FOR_TRAINING_DATA, self.PREDICT_FOR_TEST_DATA]

        def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
            """
            experiment_builder.set_predict_for_training_dataset(self.PREDICT_FOR_TRAINING_DATA.get_value(args))
            experiment_builder.set_predict_for_test_dataset(self.PREDICT_FOR_TEST_DATA.get_value(args))

    def get_extensions(self) -> List[Extension]:
        """
        May be overridden by subclasses in order to return the extensions that should be applied to the runnable.

        :return: A list that contains the extensions to be applied to the runnable
        """
        return [Runnable.PredictionDatasetExtension(), LogExtension()]

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: A `ProgramInfo` or None, if no information is provided
        """
        return None

    def run(self, args: Namespace):
        """
        Executes the runnable.

        :param args: The command line arguments specified by the user
        """
        experiment_builder = self.create_experiment_builder(args)

        for extension in self.get_extensions():
            extension.configure_experiment(args, experiment_builder)

        experiment_builder.run()

    def configure_arguments(self, cli: CommandLineInterface):
        """
        Configures the command line interface according to the extensions applied to the runnable.

        :param cli: The command line interface to be configured
        """
        for extension in self.get_extensions():
            cli.add_arguments(*extension.get_arguments())

    @abstractmethod
    def create_experiment_builder(self, args: Namespace) -> Experiment.Builder:
        """
        Must be implemented by subclasses in order to create the builder that allows to configure the experiment to be
        run by the program.

        :param args:    The command line arguments specified by the user
        :return:        The builder that has been created
        """
