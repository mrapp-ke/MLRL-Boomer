"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments.
"""
import logging as log
from argparse import ArgumentParser
from enum import Enum

from mlrl.common.config import NONE, RULE_INDUCTION_VALUES, LABEL_SAMPLING_VALUES, FEATURE_SAMPLING_VALUES, \
    INSTANCE_SAMPLING_VALUES, PARTITION_SAMPLING_VALUES, FEATURE_BINNING_VALUES, EARLY_STOPPING_VALUES, \
    RULE_PRUNING_VALUES, PARALLEL_VALUES
from mlrl.common.format import format_enum_values, format_string_set, format_dict_keys
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import SparsePolicy
from mlrl.testbed.characteristics import ARGUMENT_LABELS, ARGUMENT_LABEL_DENSITY, ARGUMENT_LABEL_SPARSITY, \
    ARGUMENT_LABEL_IMBALANCE_RATIO, ARGUMENT_LABEL_CARDINALITY, ARGUMENT_DISTINCT_LABEL_VECTORS
from mlrl.testbed.data_characteristics import ARGUMENT_EXAMPLES, ARGUMENT_FEATURES, ARGUMENT_NUMERICAL_FEATURES, \
    ARGUMENT_NOMINAL_FEATURES, ARGUMENT_FEATURE_DENSITY, ARGUMENT_FEATURE_SPARSITY
from mlrl.testbed.evaluation import ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS, \
    ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL, ARGUMENT_MICRO_F1, \
    ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION, ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, \
    ARGUMENT_MACRO_JACCARD, ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1, \
    ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS, ARGUMENT_PRECISION, ARGUMENT_RECALL, \
    ARGUMENT_F1, ARGUMENT_JACCARD, ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR, \
    ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_ABSOLUTE_PERCENTAGE_ERROR, ARGUMENT_RANK_LOSS, \
    ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION, ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, \
    ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, ARGUMENT_PREDICTION_TIME, ARGUMENT_TRAINING_TIME
from mlrl.testbed.experiments import PredictionType
from mlrl.testbed.format import ARGUMENT_DECIMALS, ARGUMENT_PERCENTAGE
from mlrl.testbed.models import ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES, \
    ARGUMENT_PRINT_NOMINAL_VALUES, ARGUMENT_PRINT_BODIES, ARGUMENT_PRINT_HEADS
from typing import Dict, Set

PARAM_LOG_LEVEL = '--log-level'

PARAM_RANDOM_STATE = '--random-state'

PARAM_DATA_DIR = '--data-dir'

PARAM_DATASET = '--dataset'

PARAM_DATA_SPLIT = '--data-split'

PARAM_PREDICTION_TYPE = '--prediction-type'

PARAM_PRINT_EVALUATION = '--print-evaluation'

PARAM_STORE_EVALUATION = '--store-evaluation'

PARAM_PRINT_PREDICTION_CHARACTERISTICS = '--print-prediction-characteristics'

PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

PARAM_PRINT_RULES = '--print-rules'

PARAM_STORE_RULES = '--store-rules'

PARAM_INCREMENTAL_EVALUATION = '--incremental-evaluation'

PARAM_EVALUATE_TRAINING_DATA = '--evaluate-training-data'

PARAM_ONE_HOT_ENCODING = '--one-hot-encoding'

PARAM_MODEL_DIR = '--model-dir'

PARAM_PARAMETER_DIR = '--parameter-dir'

PARAM_OUTPUT_DIR = '--output-dir'

PARAM_PRINT_PARAMETERS = '--print-parameters'

PARAM_STORE_PARAMETERS = '--store-parameters'

PARAM_PRINT_PREDICTIONS = '--print-predictions'

PARAM_STORE_PREDICTIONS = '--store-predictions'

PARAM_PRINT_MODEL_CHARACTERISTICS = '--print-model-characteristics'

PARAM_STORE_MODEL_CHARACTERISTICS = '--store-model-characteristics'

PARAM_FEATURE_FORMAT = '--feature-format'

PARAM_LABEL_FORMAT = '--label-format'

PARAM_PREDICTED_LABEL_FORMAT = '--predicted-label-format'

PARAM_MAX_RULES = '--max-rules'

PARAM_TIME_LIMIT = '--time-limit'

PARAM_EARLY_STOPPING = '--early-stopping'

PARAM_LABEL_SAMPLING = '--label-sampling'

PARAM_FEATURE_SAMPLING = '--feature-sampling'

PARAM_PARTITION_SAMPLING = '--holdout'

PARAM_FEATURE_BINNING = '--feature-binning'

PARAM_RULE_PRUNING = '--rule-pruning'

PARAM_RULE_MODEL_ASSEMBLAGE = '--rule-model-assemblage'

PARAM_SEQUENTIAL_POST_OPTIMIZATION = '--sequential-post-optimization'

PARAM_RULE_INDUCTION = '--rule-induction'

PARAM_PARALLEL_RULE_REFINEMENT = '--parallel-rule-refinement'

PARAM_PARALLEL_STATISTIC_UPDATE = '--parallel-statistic-update'

PARAM_PARALLEL_PREDICTION = '--parallel-prediction'

PARAM_INSTANCE_SAMPLING = '--instance-sampling'

PARAM_HEAD_TYPE = '--head-type'

DATA_SPLIT_TRAIN_TEST = 'train-test'

ARGUMENT_TEST_SIZE = 'test_size'

DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

ARGUMENT_NUM_FOLDS = 'num_folds'

ARGUMENT_CURRENT_FOLD = 'current_fold'

ARGUMENT_MIN_SIZE = 'min_size'

ARGUMENT_MAX_SIZE = 'max_size'

ARGUMENT_STEP_SIZE = 'step_size'

DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    DATA_SPLIT_TRAIN_TEST: {ARGUMENT_TEST_SIZE},
    DATA_SPLIT_CROSS_VALIDATION: {ARGUMENT_NUM_FOLDS, ARGUMENT_CURRENT_FOLD}
}

PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS,
                               ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL,
                               ARGUMENT_MICRO_F1, ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION,
                               ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, ARGUMENT_MACRO_JACCARD,
                               ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1,
                               ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS,
                               ARGUMENT_PRECISION, ARGUMENT_RECALL, ARGUMENT_F1, ARGUMENT_JACCARD,
                               ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR,
                               ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                               ARGUMENT_RANK_LOSS, ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION,
                               ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                               ARGUMENT_DECIMALS, ARGUMENT_PERCENTAGE},
    BooleanOption.FALSE.value: {}
}

STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_HAMMING_LOSS, ARGUMENT_HAMMING_ACCURACY, ARGUMENT_SUBSET_ZERO_ONE_LOSS,
                               ARGUMENT_SUBSET_ACCURACY, ARGUMENT_MICRO_PRECISION, ARGUMENT_MICRO_RECALL,
                               ARGUMENT_MICRO_F1, ARGUMENT_MICRO_JACCARD, ARGUMENT_MACRO_PRECISION,
                               ARGUMENT_MACRO_RECALL, ARGUMENT_MACRO_F1, ARGUMENT_MACRO_JACCARD,
                               ARGUMENT_EXAMPLE_WISE_PRECISION, ARGUMENT_EXAMPLE_WISE_RECALL, ARGUMENT_EXAMPLE_WISE_F1,
                               ARGUMENT_EXAMPLE_WISE_JACCARD, ARGUMENT_ACCURACY, ARGUMENT_ZERO_ONE_LOSS,
                               ARGUMENT_PRECISION, ARGUMENT_RECALL, ARGUMENT_F1, ARGUMENT_JACCARD,
                               ARGUMENT_MEAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_SQUARED_ERROR,
                               ARGUMENT_MEDIAN_ABSOLUTE_ERROR, ARGUMENT_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                               ARGUMENT_RANK_LOSS, ARGUMENT_COVERAGE_ERROR, ARGUMENT_LABEL_RANKING_AVERAGE_PRECISION,
                               ARGUMENT_DISCOUNTED_CUMULATIVE_GAIN, ARGUMENT_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN,
                               ARGUMENT_TRAINING_TIME, ARGUMENT_PREDICTION_TIME, ARGUMENT_DECIMALS,
                               ARGUMENT_PERCENTAGE},
    BooleanOption.FALSE.value: {}
}

PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_EXAMPLES, ARGUMENT_FEATURES, ARGUMENT_NUMERICAL_FEATURES,
                               ARGUMENT_NOMINAL_FEATURES, ARGUMENT_FEATURE_DENSITY, ARGUMENT_FEATURE_SPARSITY,
                               ARGUMENT_LABELS, ARGUMENT_LABEL_DENSITY, ARGUMENT_LABEL_SPARSITY,
                               ARGUMENT_LABEL_IMBALANCE_RATIO, ARGUMENT_LABEL_CARDINALITY,
                               ARGUMENT_DISTINCT_LABEL_VECTORS, ARGUMENT_DECIMALS, ARGUMENT_PERCENTAGE},
    BooleanOption.FALSE.value: {}
}

STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_LABELS, ARGUMENT_LABEL_DENSITY, ARGUMENT_LABEL_SPARSITY,
                               ARGUMENT_LABEL_IMBALANCE_RATIO, ARGUMENT_LABEL_CARDINALITY,
                               ARGUMENT_DISTINCT_LABEL_VECTORS, ARGUMENT_DECIMALS, ARGUMENT_PERCENTAGE},
    BooleanOption.FALSE.value: {}
}

STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

PRINT_RULES_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES, ARGUMENT_PRINT_NOMINAL_VALUES,
                               ARGUMENT_PRINT_BODIES, ARGUMENT_PRINT_HEADS},
    BooleanOption.FALSE.value: {}
}

STORE_RULES_VALUES = PRINT_RULES_VALUES

INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {ARGUMENT_MIN_SIZE, ARGUMENT_MAX_SIZE, ARGUMENT_STEP_SIZE},
    BooleanOption.FALSE.value: {}
}


class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARN = 'warn'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'
    NOTSET = 'notset'


def log_level(s):
    s = s.lower()
    if s == LogLevel.DEBUG.value:
        return log.DEBUG
    elif s == LogLevel.INFO.value:
        return log.INFO
    elif s == LogLevel.WARN.value or s == LogLevel.WARNING.value:
        return log.WARN
    elif s == LogLevel.ERROR.value:
        return log.ERROR
    elif s == LogLevel.CRITICAL.value or s == LogLevel.FATAL.value:
        return log.CRITICAL
    elif s == LogLevel.NOTSET.value:
        return log.NOTSET
    raise ValueError(
        'Invalid value given for parameter "' + PARAM_LOG_LEVEL + '". Must be one of ' + format_enum_values(
            LogLevel) + ', but is "' + str(s) + '".')


def boolean_string(s):
    return BooleanOption.parse(s)


def add_log_level_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LOG_LEVEL, type=log_level, default=LogLevel.INFO.value,
                        help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')


def add_learner_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_RANDOM_STATE, type=int, default=1,
                        help='The seed to be used by random number generators. Must be at least 1.')
    parser.add_argument(PARAM_DATA_DIR, type=str, required=True,
                        help='The path of the directory where the data set files are located.')
    parser.add_argument(PARAM_DATASET, type=str, required=True,
                        help='The name of the data set files without suffix.')
    parser.add_argument(PARAM_DATA_SPLIT, type=str, default=DATA_SPLIT_TRAIN_TEST,
                        help='The strategy to be used for splitting the available data into training and test sets. '
                             + 'Must be one of ' + format_dict_keys(DATA_SPLIT_VALUES) + '. For additional options '
                             + 'refer to the documentation.')
    parser.add_argument(PARAM_PRINT_EVALUATION, type=str, default=BooleanOption.TRUE.value,
                        help='Whether the evaluation results should be printed on the console or not. Must be one of '
                             + format_dict_keys(PRINT_EVALUATION_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_STORE_EVALUATION, type=str, default=BooleanOption.TRUE.value,
                        help='Whether the evaluation results should be written into output files or not. Must be one '
                             + 'of ' + format_dict_keys(STORE_EVALUATION_VALUES) + '. Does only have an effect if the '
                             + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_EVALUATE_TRAINING_DATA, type=boolean_string, default=False,
                        help='Whether the models should not only be evaluated on the test data, but also on the '
                             + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_PRINT_PREDICTION_CHARACTERISTICS, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of binary predictions should be printed on the console or '
                             + 'not. Must be one of ' + format_dict_keys(PRINT_PREDICTION_CHARACTERISTICS_VALUES) + '. '
                             + 'Does only have an effect if the parameter ' + PARAM_PREDICTION_TYPE + ' is set to '
                             + PredictionType.LABELS.value + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_STORE_PREDICTION_CHARACTERISTICS, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of binary predictions should be written into output files or '
                             + 'not. Must be one of ' + format_dict_keys(STORE_PREDICTION_CHARACTERISTICS_VALUES) + '. '
                             + 'Does only have an effect if the parameter ' + PARAM_PREDICTION_TYPE + ' is set to '
                             + PredictionType.LABELS.value + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PRINT_DATA_CHARACTERISTICS, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of the training data should be printed on the console or '
                             + 'not. Must be one of ' + format_dict_keys(PRINT_DATA_CHARACTERISTICS_VALUES) + '. For '
                             + 'additional options refer to the documentation.')
    parser.add_argument(PARAM_STORE_DATA_CHARACTERISTICS, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of the training data should be written into output files or '
                             + 'not. Must be one of ' + format_dict_keys(STORE_DATA_CHARACTERISTICS_VALUES) + '. Does '
                             + 'only have an effect if the parameter ' + PARAM_OUTPUT_DIR + ' is specified. For '
                             + 'additional options refer to the documentation')
    parser.add_argument(PARAM_ONE_HOT_ENCODING, type=boolean_string, default=False,
                        help='Whether one-hot-encoding should be used to encode nominal attributes or not. Must be one '
                             + 'of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_MODEL_DIR, type=str,
                        help='The path of the directory where models should be stored.')
    parser.add_argument(PARAM_PARAMETER_DIR, type=str,
                        help='The path of the directory where configuration files, which specify the parameters to be '
                             + 'used by the algorithm, are located.')
    parser.add_argument(PARAM_OUTPUT_DIR, type=str,
                        help='The path of the directory where experimental results should be saved.')
    parser.add_argument(PARAM_PRINT_PARAMETERS, type=boolean_string, default=False,
                        help='Whether the parameter setting should be printed on the console or not. Must be one of '
                             + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_PARAMETERS, type=boolean_string, default=False,
                        help='Whether the parameter setting should be written into output files or not. Must be one of '
                             + format_enum_values(BooleanOption) + '. Does only have an effect, if the parameter '
                             + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_PREDICTIONS, type=boolean_string, default=False,
                        help='Whether the predictions for individual examples and labels should be printed on the ' +
                             'console or not. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_PREDICTIONS, type=boolean_string, default=False,
                        help='Whether the predictions for individual examples and labels should be written into output '
                             + 'files or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does only have '
                             + 'an effect, if the parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PREDICTION_TYPE, type=str, default=PredictionType.LABELS.value,
                        help='The type of predictions that should be obtained from the learner. Must be one of '
                             + format_enum_values(PredictionType) + '.')


def add_rule_learner_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_INCREMENTAL_EVALUATION, type=str, default=BooleanOption.FALSE.value,
                        help='Whether models should be evaluated repeatedly, using only a subset of the induced rules '
                             + 'with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption)
                             + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PRINT_MODEL_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of models should be printed on the console or not. Must be '
                             + 'one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_MODEL_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of models should be written into output files or not. Must '
                             + 'be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if the '
                             + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_RULES, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the induced rules should be printed on the console or not. Must be one of '
                             + format_dict_keys(PRINT_RULES_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_STORE_RULES, type=str, default=BooleanOption.FALSE.value,
                        help='Whether the induced rules should be written into a text file or not. Must be one of '
                             + format_dict_keys(STORE_RULES_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the documentation.')
    parser.add_argument(PARAM_FEATURE_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the feature matrix. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_LABEL_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the label matrix. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_PREDICTED_LABEL_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of predicted labels. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')


def add_max_rules_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_MAX_RULES, type=int,
                        help='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of '
                             + 'rules should not be restricted.')


def add_time_limit_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_TIME_LIMIT, type=int,
                        help='The duration in seconds after which the induction of rules should be canceled. Must be '
                             + 'at least 1 or 0, if no time limit should be set.')


def add_sequential_post_optimization_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_SEQUENTIAL_POST_OPTIMIZATION, type=str,
                        help='Whether each rule in a previously learned model should be optimized by being relearned '
                             + 'in the context of the other rules. Must be one of ' + format_enum_values(BooleanOption)
                             + '. For additional options refer to the documentation.')


def add_label_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LABEL_SAMPLING, type=str,
                        help='The name of the strategy to be used for label sampling. Must be one of '
                             + format_dict_keys(LABEL_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_instance_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_INSTANCE_SAMPLING, type=str,
                        help='The name of the strategy to be used for instance sampling. Must be one of'
                             + format_dict_keys(INSTANCE_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_feature_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_FEATURE_SAMPLING, type=str,
                        help='The name of the strategy to be used for feature sampling. Must be one of '
                             + format_dict_keys(FEATURE_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_partition_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARTITION_SAMPLING, type=str,
                        help='The name of the strategy to be used for creating a holdout set. Must be one of '
                             + format_dict_keys(PARTITION_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_feature_binning_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_FEATURE_BINNING, type=str,
                        help='The name of the strategy to be used for feature binning. Must be one of '
                             + format_dict_keys(FEATURE_BINNING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_early_stopping_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_EARLY_STOPPING, type=str,
                        help='The name of the strategy to be used for early stopping. Must be one of '
                             + format_dict_keys(EARLY_STOPPING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')


def add_pruning_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_RULE_PRUNING, type=str,
                        help='The name of the strategy to be used for pruning individual rules. Must be one of '
                             + format_string_set(RULE_PRUNING_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_INSTANCE_SAMPLING + ' is not set to "none".')


def add_rule_induction_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_RULE_INDUCTION, type=str,
                        help='The name of the algorithm to be used for the induction of individual rules. Must be one '
                             + 'of ' + format_string_set(RULE_INDUCTION_VALUES) + '. For additional options refer to '
                             + 'the documentation')


def add_parallel_prediction_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_PREDICTION, type=str,
                        help='Whether predictions for different examples should be obtained in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to '
                             + 'the documentation.')


def add_parallel_rule_refinement_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=str,
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to '
                             + 'the documentation.')


def add_parallel_statistic_update_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=str,
                        help='Whether the confusion matrices for different examples should be calculated in parallel '
                             + 'or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional '
                             + 'options refer to the documentation')
