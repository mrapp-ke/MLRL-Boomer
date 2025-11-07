"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import List, Sequence, Set, override

from mlrl.testbed.command import ArgumentList
from mlrl.testbed.experiments.input.dataset.arguments import DatasetArguments
from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.extension import InputExtension
from mlrl.testbed.experiments.input.sources import FileSource, Source
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode

from mlrl.util.cli import Argument, PathArgument


class DatasetExtension(Extension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets.
    """

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(InputExtension(), *dependencies)

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {DatasetArguments.DATASET_NAME}

    @abstractmethod
    def create_sources(self, dataset: InputDataset, args: Namespace) -> Sequence[Source]:
        """
        Creates and returns one or several sources, the dataset should be loaded from.

        :param dataset: The dataset that should be loaded
        :param args:    The command line arguments specified by the user
        :return:        A sequence that contains the sources, the dataset should be loaded from
        """

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE}


class DatasetFileExtension(DatasetExtension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets from files.
    """

    DATASET_DIRECTORY = PathArgument(
        '--data-dir',
        required=True,
        description='The path to the directory where the dataset files are located.',
    )

    @override
    def _get_arguments(self, mode: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return super()._get_arguments(mode) | {self.DATASET_DIRECTORY}

    @override
    def create_sources(self, dataset: InputDataset, args: Namespace) -> Sequence[Source]:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.extension.DatasetExtension.create_sources`
        """
        dataset_directory = self.DATASET_DIRECTORY.get_value(args)
        return self._create_file_sources(dataset_directory, dataset, args)

    @abstractmethod
    def _create_file_sources(self, dataset_directory: Path, dataset: InputDataset,
                             args: Namespace) -> Sequence[FileSource]:
        """
        Must be implemented by subclasses in order to create one or several sources, the dataset should be loaded from.

        :param dataset_directory:   The path to the directory, the dataset should be loaded from
        :param dataset:             The dataset that should be loaded
        :param args:                The command line arguments specified by the user
        :return:                    A list that contains the sources, the dataset should be loaded from
        """

    @staticmethod
    def parse_dataset_args_from_config(config: BatchMode.ConfigFile) -> List[ArgumentList]:
        """
        Parses and returns the command line arguments for using the datasets specified in a configuration file.

        :param config:  The configuration file that should be parsed
        :return:        A list that contains the command line arguments that have been parsed
        """
        datasets = config.yaml_dict.get('datasets', [])

        if not datasets:
            raise ValueError('No datasets are specified in the configuration file "' + str(config) + '"')

        dataset_args = []

        for dataset_dict in datasets:
            names = dataset_dict['names']

            if isinstance(names, str):
                dataset_names = [names]
            else:
                dataset_names = list(names)

            for dataset_name in dataset_names:
                dataset_args.append(
                    ArgumentList([
                        DatasetFileExtension.DATASET_DIRECTORY.name, dataset_dict['directory'],
                        DatasetArguments.DATASET_NAME.name, dataset_name
                    ]))

        return dataset_args
