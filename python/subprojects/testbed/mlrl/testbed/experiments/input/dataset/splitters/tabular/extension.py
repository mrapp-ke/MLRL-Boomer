"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to split datasets into training and test datasets.
"""
from argparse import Namespace
from typing import Set

from mlrl.common.config.parameters import NONE

from mlrl.testbed.experiments.input.dataset.preprocessors.tabular.extension import PreprocessorExtension
from mlrl.testbed.experiments.input.dataset.splitters.extension import DatasetFileExtension
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.input.dataset.splitters.splitter_no import NoSplitter
from mlrl.testbed.experiments.input.dataset.splitters.tabular.splitter_bipartition import BipartitionSplitter
from mlrl.testbed.experiments.input.dataset.splitters.tabular.splitter_cross_validation import CrossValidationSplitter
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, IntArgument, SetArgument
from mlrl.util.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal

VALUE_TRAIN_TEST = 'train-test'

OPTION_TEST_SIZE = 'test_size'

VALUE_CROSS_VALIDATION = 'cross-validation'

OPTION_NUM_FOLDS = 'num_folds'

OPTION_FIRST_FOLD = 'last_fold'

OPTION_LAST_FOLD = 'first_fold'


class DatasetSplitterExtension(Extension):
    """
    An extension that configures the functionality to split datasets into training and test datasets.
    """

    RANDOM_STATE = IntArgument(
        '--random-state',
        description='The seed to be used by random number generators. Must be at least 1.',
        default=1,
    )

    DATASET_SPLITTER = SetArgument(
        '--data-split',
        description='The strategy to be used for splitting the available data into training and test sets.',
        default=VALUE_TRAIN_TEST,
        values={
            NONE: {},
            VALUE_TRAIN_TEST: {OPTION_TEST_SIZE},
            VALUE_CROSS_VALIDATION: {OPTION_NUM_FOLDS, OPTION_FIRST_FOLD, OPTION_LAST_FOLD}
        },
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(PreprocessorExtension(), DatasetFileExtension(), *dependencies)

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.RANDOM_STATE, self.DATASET_SPLITTER}

    @staticmethod
    def get_random_state(args: Namespace) -> int:
        """
        Returns the seed to be used by random number generators.

        :param args:    The command line arguments specified by the user
        :return:        The seed to be used
        """
        random_state = DatasetSplitterExtension.RANDOM_STATE.get_value(args)
        assert_greater_or_equal(DatasetSplitterExtension.RANDOM_STATE.name, random_state, 1)
        return random_state

    @staticmethod
    def get_dataset_splitter(args: Namespace) -> DatasetSplitter:
        """
        Returns the `DatasetSplitter` to be used for splitting datasets into training and test datasets according to the
        configuration.

        :param args:    The command line arguments specified by the user
        :return:        The `DatasetSplitter` to be used
        """
        dataset_reader = DatasetFileExtension.get_dataset_reader(args)
        dataset_reader.add_preprocessors(*PreprocessorExtension.get_preprocessors(args))
        value, options = DatasetSplitterExtension.DATASET_SPLITTER.get_value(args)

        if value == VALUE_CROSS_VALIDATION:
            num_folds = options.get_int(OPTION_NUM_FOLDS, 10)
            assert_greater_or_equal(OPTION_NUM_FOLDS, num_folds, 2)
            first_fold = options.get_int(OPTION_FIRST_FOLD, 1)
            assert_greater_or_equal(OPTION_FIRST_FOLD, first_fold, 1)
            assert_less_or_equal(OPTION_FIRST_FOLD, first_fold, num_folds)
            last_fold = options.get_int(OPTION_LAST_FOLD, num_folds)
            assert_greater_or_equal(OPTION_LAST_FOLD, last_fold, first_fold)
            assert_less_or_equal(OPTION_LAST_FOLD, last_fold, num_folds)
            return CrossValidationSplitter(dataset_reader,
                                           num_folds=num_folds,
                                           first_fold=first_fold - 1,
                                           last_fold=last_fold,
                                           random_state=DatasetSplitterExtension.get_random_state(args))
        if value == VALUE_TRAIN_TEST:
            test_size = options.get_float(OPTION_TEST_SIZE, 0.33)
            assert_greater(OPTION_TEST_SIZE, test_size, 0)
            assert_less(OPTION_TEST_SIZE, test_size, 1)
            return BipartitionSplitter(dataset_reader,
                                       test_size=test_size,
                                       random_state=DatasetSplitterExtension.get_random_state(args))

        return NoSplitter(dataset_reader)
