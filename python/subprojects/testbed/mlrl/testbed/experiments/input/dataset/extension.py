"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Set

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.reader import DatasetReader
from mlrl.testbed.experiments.input.sources import FileSource, Source
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, StringArgument


class DatasetExtension(Extension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets.
    """

    DATASET_NAME = StringArgument(
        '--dataset',
        required=True,
        description='The name of the dataset.',
    )

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.DATASET_NAME}

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
        dataset = InputDataset(name=self.DATASET_NAME.get_value(args))
        source = self._create_source(dataset, args)
        return DatasetReader(source=source, input_data=dataset)


class DatasetFileExtension(DatasetExtension, ABC):
    """
    An abstract base class for all extensions that configure the functionality to load datasets from files.
    """

    DATASET_DIRECTORY = StringArgument(
        '--data-dir',
        required=True,
        description='The path to the directory where the dataset files are located.',
    )

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return super()._get_arguments() | {self.DATASET_DIRECTORY}

    def _create_source(self, dataset: InputDataset, args: Namespace) -> Source:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.extension.DatasetExtension._create_source`
        """
        dataset_directory = self.DATASET_DIRECTORY.get_value(args)
        return self._create_file_source(dataset_directory, dataset, args)

    @abstractmethod
    def _create_file_source(self, dataset_directory: str, dataset: InputDataset, args: Namespace) -> FileSource:
        """
        Must be implemented by subclasses in order to create the `FileSource`, the dataset should be loaded from.

        :param dataset_directory:   The path to the directory, the dataset should be loaded from
        :param dataset:             The dataset that should be loaded
        :param args:                The command line arguments specified by the user
        :return:                    The `FileSource`, the dataset should be loaded from
        """
