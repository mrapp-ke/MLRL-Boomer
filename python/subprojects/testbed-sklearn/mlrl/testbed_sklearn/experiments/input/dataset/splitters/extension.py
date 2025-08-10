"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to split datasets into training and test datasets.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed_sklearn.experiments.input.dataset.extension import ArffFileExtension
from mlrl.testbed_sklearn.experiments.input.dataset.preprocessors.extension import PreprocessorExtension
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.splitter_bipartition import BipartitionSplitter
from mlrl.testbed_sklearn.experiments.input.dataset.splitters.splitter_cross_validation import CrossValidationSplitter

from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_no import NoSplitter
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument
from mlrl.util.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal


class DatasetSplitterExtension(Extension):
    """
    An extension that configures the functionality to split tabular datasets into training and test datasets.
    """

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(PreprocessorExtension(), ArffFileExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {DatasetSplitterArguments.RANDOM_STATE, DatasetSplitterArguments.DATASET_SPLITTER}

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
    def get_dataset_splitter(args: Namespace) -> DatasetSplitter:
        """
        Returns the `DatasetSplitter` to be used for splitting datasets into training and test datasets according to the
        configuration.

        :param args:    The command line arguments specified by the user
        :return:        The `DatasetSplitter` to be used
        """
        dataset_reader = ArffFileExtension().get_dataset_reader(args)
        dataset_reader.add_preprocessors(*PreprocessorExtension.get_preprocessors(args))
        dataset_splitter, options = DatasetSplitterArguments.DATASET_SPLITTER.get_value(args)

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
