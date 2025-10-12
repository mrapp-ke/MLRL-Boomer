"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing different modes of operation.
"""
import logging as log

from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import List, override

from mlrl.testbed.experiments.input.meta_data.meta_data import InputMetaData
from mlrl.testbed.experiments.input.meta_data.reader import MetaDataReader
from mlrl.testbed.experiments.input.sources import YamlFileSource
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.cli import Argument, CommandLineInterface, PathArgument, SetArgument


class Mode(ABC):
    """
    An abstract base class for all modes of operation.
    """

    MODE_SINGLE = 'single'

    MODE_BATCH = 'batch'

    MODE_READ = 'read'

    MODE_RUN = 'run'

    MODE = SetArgument(
        '--mode',
        values={MODE_SINGLE, MODE_BATCH, MODE_READ, MODE_RUN},
        description='The mode of operation to be used.',
        default=MODE_SINGLE,
    )

    @abstractmethod
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        """
        Must be implemented by subclasses in order to configure the command line interface according to the mode of
        operation.

        :param cli:                     The command line interface to be configured
        :param extension_arguments:     The arguments that should be added to the command line interface according to
                                        the registered extensions
        :param algorithmic_arguments:   The arguments that should be added to the command line interface for configuring
                                        the algorithm's hyperparameters
        """

    @abstractmethod
    def run_experiment(self, args: Namespace, recipe: Recipe):
        """
        Must be implemented by subclasses in order to run an experiment according to the command line arguments
        specified by the user.

        :param args:    The command line arguments specified by the user
        :param recipe:  A `Recipe` that provides access to the ingredients that are needed for setting up experiments
        """


class InputMode(Mode, ABC):
    """
    An abstract base class for all modes of operation that access the meta-data of an earlier experiment.
    """

    INPUT_DIR = PathArgument(
        '--input-dir',
        required=True,
        description='An absolute or relative path to a directory that contains a metadata.yml file that has been '
        + 'created when running one or several experiments in the past.',
    )

    @override
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        cli.add_arguments(self.INPUT_DIR, *extension_arguments)

    @staticmethod
    def __read_meta_data(args: Namespace, recipe: Recipe, input_directory: Path) -> MetaData:
        log.info('Reading meta-data...')
        problem_domain = recipe.create_problem_domain(args)
        state = ExperimentState(meta_data=MetaData(), problem_domain=problem_domain)
        reader = MetaDataReader(
            YamlFileSource(directory=input_directory, schema_file_path=InputMetaData.SCHEMA_FILE_PATH))
        meta_data = reader.read(state).meta_data
        log.info('Successfully read meta-data')
        return meta_data

    @staticmethod
    def __check_version(meta_data: MetaData):
        log.debug('Checking for version conflicts...')
        meta_data_version = meta_data.version
        current_version = MetaData().version
        log.debug(
            'Experimental results have been created with version "%s" of the package "mlrl-testbed", version "%s" is '
            + 'currently used', meta_data_version, current_version)

        if meta_data_version > current_version:
            log.warning(
                'Experimental results have been created with a version (%s) of the package "mlrl-testbed" that is '
                + 'greater than currently used (%s).', meta_data_version, current_version)
        else:
            log.debug('No version conflicts detected')

    @override
    def run_experiment(self, args: Namespace, recipe: Recipe):
        input_directory = self.INPUT_DIR.get_value(args)

        if input_directory:
            meta_data = self.__read_meta_data(args, recipe, input_directory)
            self.__check_version(meta_data)
            self._run_experiment(args, recipe, meta_data, input_directory)

    @abstractmethod
    def _run_experiment(self, args: Namespace, recipe: Recipe, meta_data: MetaData, input_directory: Path):
        """
        Must be implemented by subclasses in order to run an experiment that accesses the meta-data of an earlie
        experiment according to the command line arguments specified by the user.

        :param args:            The command line arguments specified by the user
        :param recipe:          A `Recipe` that provides access to the ingredients that are needed for setting up
                                experiments
        :param meta_data:       The `MetaData` of the earlier experiment
        :param input_directory: The path to the input directory, the meta-data has been read from
        """
