"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to split datasets into training and test datasets.
"""
from mlrl.util.cli import NONE, IntArgument, SetArgument


class DatasetSplitterArguments:
    """
    Defines command line arguments for configuring the functionality to split datasets into training and test datasets.
    """

    RANDOM_STATE = IntArgument(
        '--random-state',
        description='The seed to be used by random number generators. Must be at least 1.',
        default=1,
    )

    VALUE_TRAIN_TEST = 'train-test'

    OPTION_TEST_SIZE = 'test_size'

    VALUE_CROSS_VALIDATION = 'cross-validation'

    OPTION_NUM_FOLDS = 'num_folds'

    OPTION_FIRST_FOLD = 'first_fold'

    OPTION_LAST_FOLD = 'last_fold'

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
