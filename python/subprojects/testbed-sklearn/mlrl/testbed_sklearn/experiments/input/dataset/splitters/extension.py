"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to split datasets into training and test datasets.
"""
from argparse import Namespace
from itertools import chain
from typing import List, Set, override

from mlrl.testbed_sklearn.experiments.input.dataset.extension import ArffFileExtension, SvmFileExtension
from mlrl.testbed_sklearn.experiments.input.dataset.preprocessors.extension import PreprocessorExtension
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.splitter_bipartition import BipartitionSplitter
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.splitter_cross_validation import CrossValidationSplitter

from mlrl.testbed.experiments.input.dataset.arguments import DatasetArguments
from mlrl.testbed.experiments.input.dataset.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.extension import DatasetFileExtension
from mlrl.testbed.experiments.input.dataset.reader import DatasetReader
from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_no import NoSplitter
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import AUTO, Argument, SetArgument
from mlrl.util.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal


class DatasetSplitterExtension(Extension):
    """
    An extension that configures the functionality to split tabular datasets into training and test datasets.
    """

    DATASET_READER_EXTENSIONS: List[DatasetFileExtension] = [ArffFileExtension(), SvmFileExtension()]

    DATASET_FORMAT = SetArgument('--dataset-format',
                                 default=AUTO,
                                 values={AUTO}
                                 | set(map(lambda extension: extension.file_type, DATASET_READER_EXTENSIONS)),
                                 description='The dataset format to be used.')

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(PreprocessorExtension(), *self.DATASET_READER_EXTENSIONS, *dependencies)

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {DatasetSplitterArguments.RANDOM_STATE, DatasetSplitterArguments.DATASET_SPLITTER, self.DATASET_FORMAT}

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.BATCH}

    @staticmethod
    def get_random_state(args: Namespace) -> int:
        """
        Returns the seed to be used by random number generators.

        :param args:    The command line arguments specified by the user
        :return:        The seed to be used
        """
        random_state = DatasetSplitterArguments.RANDOM_STATE.get_value(args)
        assert_greater_or_equal(DatasetSplitterArguments.RANDOM_STATE.name, random_state, 1)
        return random_state

    @staticmethod
    def get_dataset_splitter(args: Namespace, load_dataset: bool = True) -> DatasetSplitter:
        """
        Returns the `DatasetSplitter` to be used for splitting datasets into training and test datasets according to the
        configuration.

        :param args:            The command line arguments specified by the user
        :param load_dataset:    True, if the dataset should be loaded, False otherwise
        :return:                The `DatasetSplitter` to be used
        """
        if load_dataset:
            dataset = InputDataset(name=DatasetArguments.DATASET_NAME.get_value(args))
            dataset_format = DatasetSplitterExtension.DATASET_FORMAT.get_value(args)
            sources = chain.from_iterable(
                extension.create_sources(dataset, args)
                for extension in DatasetSplitterExtension.DATASET_READER_EXTENSIONS
                if dataset_format in {AUTO, extension.file_type})
            dataset_reader = DatasetReader(dataset, *sources)
            dataset_reader.add_preprocessors(*PreprocessorExtension.get_preprocessors(args))
        else:
            dataset_reader = None

        dataset_splitter, options = DatasetSplitterArguments.DATASET_SPLITTER.get_value_and_options(args)

        if dataset_splitter == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
            num_folds = options.get_int(DatasetSplitterArguments.OPTION_NUM_FOLDS, 10)
            assert_greater_or_equal(DatasetSplitterArguments.OPTION_NUM_FOLDS, num_folds, 2)
            first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 1)
            assert_greater_or_equal(DatasetSplitterArguments.OPTION_FIRST_FOLD, first_fold, 1)
            assert_less_or_equal(DatasetSplitterArguments.OPTION_FIRST_FOLD, first_fold, num_folds)
            last_fold = options.get_int(DatasetSplitterArguments.OPTION_LAST_FOLD, num_folds)
            assert_greater_or_equal(DatasetSplitterArguments.OPTION_LAST_FOLD, last_fold, first_fold)
            assert_less_or_equal(DatasetSplitterArguments.OPTION_LAST_FOLD, last_fold, num_folds)
            return CrossValidationSplitter(dataset_reader,
                                           num_folds=num_folds,
                                           first_fold=first_fold - 1,
                                           last_fold=last_fold,
                                           random_state=DatasetSplitterExtension.get_random_state(args))
        if dataset_splitter == DatasetSplitterArguments.VALUE_TRAIN_TEST:
            test_size = options.get_float(DatasetSplitterArguments.OPTION_TEST_SIZE, 0.33)
            assert_greater(DatasetSplitterArguments.OPTION_TEST_SIZE, test_size, 0)
            assert_less(DatasetSplitterArguments.OPTION_TEST_SIZE, test_size, 1)
            return BipartitionSplitter(dataset_reader,
                                       test_size=test_size,
                                       random_state=DatasetSplitterExtension.get_random_state(args))

        return NoSplitter(dataset_reader)
