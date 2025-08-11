"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import List, Set, Type, override

from mlrl.testbed.command import ArgumentList
from mlrl.testbed.experiments.input.dataset.arguments import DatasetArguments
from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.reader import DatasetReader
from mlrl.testbed.experiments.input.sources import FileSource, Source
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import Mode, SingleMode
from mlrl.testbed.modes.mode_batch import BatchMode

from mlrl.util.cli import Argument, StringArgument


class DatasetExtension(Extension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets.
    """

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {DatasetArguments.DATASET_NAME}

    @abstractmethod
    def _create_source(self, dataset: InputDataset, args: Namespace) -> Source:
        """
        Must be implemented by subclasses in order to create the `Source`, the dataset should be loaded from.

        :param dataset: The dataset that should be loaded
        :param args:    The command line arguments specified by the user
        :return:        The `Source`, the dataset should be loaded from
        """

    def get_dataset_reader(self, args: Namespace) -> DatasetReader:
        """
        Returns the `DatasetReader` to be used for loading datasets according to the configuration.

        :param args:    The command line arguments specified by the user
        :return:        The `DatasetReader` to be used
        """
        dataset = InputDataset(name=DatasetArguments.DATASET_NAME.get_value(args))
        source = self._create_source(dataset, args)
        return DatasetReader(source=source, input_data=dataset)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode}


class DatasetFileExtension(DatasetExtension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets from files.
    """

    DATASET_DIRECTORY = StringArgument(
        '--data-dir',
        required=True,
        description='The path to the directory where the dataset files are located.',
    )

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return super()._get_arguments() | {self.DATASET_DIRECTORY}

    @override
    def _create_source(self, dataset: InputDataset, args: Namespace) -> Source:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.extension.DatasetExtension._create_source`
        """
        dataset_directory = self.DATASET_DIRECTORY.get_value(args)
        return self._create_file_source(Path(dataset_directory), dataset, args)

    @abstractmethod
    def _create_file_source(self, dataset_directory: Path, dataset: InputDataset, args: Namespace) -> FileSource:
        """
        Must be implemented by subclasses in order to create the `FileSource`, the dataset should be loaded from.

        :param dataset_directory:   The path to the directory, the dataset should be loaded from
        :param dataset:             The dataset that should be loaded
        :param args:                The command line arguments specified by the user
        :return:                    The `FileSource`, the dataset should be loaded from
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
