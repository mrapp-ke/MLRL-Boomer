"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for re-running experiments based on their meta-data.
"""
import logging as log
import subprocess

from argparse import Namespace
from pathlib import Path
from typing import List, override

from mlrl.testbed.command import ArgumentDict, Command
from mlrl.testbed.experiments.input.meta_data.meta_data import InputMetaData
from mlrl.testbed.experiments.input.meta_data.reader import MetaDataReader
from mlrl.testbed.experiments.input.sources import YamlFileSource
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import Argument, CommandLineInterface, PathArgument
from mlrl.util.format import format_iterable


class RunMode(Mode):
    """
    An abstract base class for all modes of operation for re-running experiments based on their meta-data.
    """

    INPUT_DIR = PathArgument(
        '--input-dir',
        required=True,
        description='An absolute or relative path to a directory that contains a metadata.yml file that has been '
        + 'created when running one or several experiments in the past.',
    )

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
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        cli.add_arguments(self.INPUT_DIR, *extension_arguments)

    @override
    def run_experiment(self, args: Namespace, recipe: Recipe):
        input_directory = self.INPUT_DIR.get_value(args)

        if input_directory:
            meta_data = self.__read_meta_data(args, recipe, input_directory)
            self.__check_version(meta_data)

            meta_data_command = meta_data.command
            log.info('Re-running experiment "%s"...', meta_data_command)
            overridden_arguments = Command.from_argv().argument_list.filter(*self.INPUT_DIR.names, *Mode.MODE.names)

            if overridden_arguments:
                meta_data_arguments = meta_data_command.argument_dict
                merged_arguments = meta_data_arguments | overridden_arguments.to_dict()
                meta_data_command = Command.from_dict(meta_data_command.module_name, ArgumentDict(merged_arguments))
                log.info(
                    'Arguments "%s" modifying the original experiments have been provided. The resulting command is '
                    + '"%s"...', format_iterable(overridden_arguments, separator=' '), meta_data_command)

            log.info('')
            subprocess.run(list(meta_data_command), check=False)
