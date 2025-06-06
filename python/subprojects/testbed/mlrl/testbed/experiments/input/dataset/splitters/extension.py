"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to load datasets.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed_arff.experiments.input.sources.source_arff import ArffFileSource

from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.reader import DatasetReader
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, StringArgument


class DatasetFileExtension(Extension):
    """
    An extension that configures the functionality to load datasets from files.
    """

    DATASET_DIRECTORY = StringArgument(
        '--data-dir',
        required=True,
        description='The path to the directory where the dataset files are located.',
    )

    DATASET_NAME = StringArgument(
        '--dataset',
        required=True,
        description='The name of the dataset files without suffix.',
    )

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.DATASET_DIRECTORY, self.DATASET_NAME}

    @staticmethod
    def get_dataset_reader(args: Namespace) -> DatasetReader:
        """
        Returns the `DatasetReader` to be used for loading datasets according to the configuration.

        :param args:    The command line arguments specified by the user
        :return:        The `DatasetReader` to be used
        """
        dataset = InputDataset(name=DatasetFileExtension.DATASET_NAME.get_value(args))
        source = ArffFileSource(directory=args.data_dir)
        return DatasetReader(source=source, input_data=dataset)
