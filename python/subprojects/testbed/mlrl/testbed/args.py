"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments.
"""
import logging as log
from argparse import ArgumentParser
from enum import Enum

from mlrl.common.config import NONE, RULE_INDUCTION_VALUES, LABEL_SAMPLING_VALUES, FEATURE_SAMPLING_VALUES, \
    INSTANCE_SAMPLING_VALUES, PARTITION_SAMPLING_VALUES, FEATURE_BINNING_VALUES, GLOBAL_PRUNING_VALUES, \
    RULE_PRUNING_VALUES, PARALLEL_VALUES
from mlrl.common.format import format_enum_values, format_string_set, format_dict_keys
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import SparsePolicy
from mlrl.testbed.characteristics import OPTION_LABELS, OPTION_LABEL_DENSITY, OPTION_LABEL_SPARSITY, \
    OPTION_LABEL_IMBALANCE_RATIO, OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS
from mlrl.testbed.data_characteristics import OPTION_EXAMPLES, OPTION_FEATURES, OPTION_NUMERICAL_FEATURES, \
    OPTION_NOMINAL_FEATURES, OPTION_FEATURE_DENSITY, OPTION_FEATURE_SPARSITY
from mlrl.testbed.evaluation import OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, \
    OPTION_SUBSET_ZERO_ONE_LOSS, OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, \
    OPTION_MICRO_JACCARD, OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD, \
    OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1, OPTION_EXAMPLE_WISE_JACCARD, \
    OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL, OPTION_F1, OPTION_JACCARD, \
    OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR, OPTION_MEDIAN_ABSOLUTE_ERROR, \
    OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS, OPTION_COVERAGE_ERROR, \
    OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN, \
    OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_PREDICTION_TIME, OPTION_TRAINING_TIME
from mlrl.testbed.experiments import PredictionType
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE
from mlrl.testbed.models import OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_LABEL_NAMES, OPTION_PRINT_NOMINAL_VALUES, \
    OPTION_PRINT_BODIES, OPTION_PRINT_HEADS
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

PARAM_PREDICTION_FORMAT = '--prediction-format'

PARAM_MAX_RULES = '--max-rules'

PARAM_TIME_LIMIT = '--time-limit'

PARAM_LABEL_SAMPLING = '--label-sampling'

PARAM_FEATURE_SAMPLING = '--feature-sampling'

PARAM_PARTITION_SAMPLING = '--holdout'

PARAM_FEATURE_BINNING = '--feature-binning'

PARAM_RULE_PRUNING = '--rule-pruning'

PARAM_GLOBAL_PRUNING = '--global-pruning'

PARAM_RULE_MODEL_ASSEMBLAGE = '--rule-model-assemblage'

PARAM_SEQUENTIAL_POST_OPTIMIZATION = '--sequential-post-optimization'

PARAM_RULE_INDUCTION = '--rule-induction'

PARAM_PARALLEL_RULE_REFINEMENT = '--parallel-rule-refinement'

PARAM_PARALLEL_STATISTIC_UPDATE = '--parallel-statistic-update'

PARAM_PARALLEL_PREDICTION = '--parallel-prediction'

PARAM_INSTANCE_SAMPLING = '--instance-sampling'

PARAM_HEAD_TYPE = '--head-type'

DATA_SPLIT_TRAIN_TEST = 'train-test'

OPTION_TEST_SIZE = 'test_size'

DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

OPTION_NUM_FOLDS = 'num_folds'

OPTION_CURRENT_FOLD = 'current_fold'

OPTION_MIN_SIZE = 'min_size'

OPTION_MAX_SIZE = 'max_size'

OPTION_STEP_SIZE = 'step_size'

DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    DATA_SPLIT_TRAIN_TEST: {OPTION_TEST_SIZE},
    DATA_SPLIT_CROSS_VALIDATION: {OPTION_NUM_FOLDS, OPTION_CURRENT_FOLD}
}

PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {
        OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
        OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
        OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
        OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1, OPTION_EXAMPLE_WISE_JACCARD,
        OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL, OPTION_F1, OPTION_JACCARD,
        OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR, OPTION_MEDIAN_ABSOLUTE_ERROR,
        OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS, OPTION_COVERAGE_ERROR,
        OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
        OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_DECIMALS, OPTION_PERCENTAGE
    },
    BooleanOption.FALSE.value: {}
}

STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {
        OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
        OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
        OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
        OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1, OPTION_EXAMPLE_WISE_JACCARD,
        OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL, OPTION_F1, OPTION_JACCARD,
        OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR, OPTION_MEDIAN_ABSOLUTE_ERROR,
        OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS, OPTION_COVERAGE_ERROR,
        OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
        OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_TRAINING_TIME, OPTION_PREDICTION_TIME, OPTION_DECIMALS,
        OPTION_PERCENTAGE
    },
    BooleanOption.FALSE.value: {}
}

PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {
        OPTION_EXAMPLES, OPTION_FEATURES, OPTION_NUMERICAL_FEATURES, OPTION_NOMINAL_FEATURES, OPTION_FEATURE_DENSITY,
        OPTION_FEATURE_SPARSITY, OPTION_LABELS, OPTION_LABEL_DENSITY, OPTION_LABEL_SPARSITY,
        OPTION_LABEL_IMBALANCE_RATIO, OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS,
        OPTION_PERCENTAGE
    },
    BooleanOption.FALSE.value: {}
}

STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {
        OPTION_LABELS, OPTION_LABEL_DENSITY, OPTION_LABEL_SPARSITY, OPTION_LABEL_IMBALANCE_RATIO,
        OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS, OPTION_PERCENTAGE
    },
    BooleanOption.FALSE.value: {}
}

STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

PRINT_RULES_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {
        OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_LABEL_NAMES, OPTION_PRINT_NOMINAL_VALUES, OPTION_PRINT_BODIES,
        OPTION_PRINT_HEADS
    },
    BooleanOption.FALSE.value: {}
}

STORE_RULES_VALUES = PRINT_RULES_VALUES

INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
    BooleanOption.TRUE.value: {OPTION_MIN_SIZE, OPTION_MAX_SIZE, OPTION_STEP_SIZE},
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
    raise ValueError('Invalid value given for parameter "' + PARAM_LOG_LEVEL + '". Must be one of '
                     + format_enum_values(LogLevel) + ', but is "' + str(s) + '".')


def boolean_string(s):
    return BooleanOption.parse(s)


def add_log_level_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LOG_LEVEL,
                        type=log_level,
                        default=LogLevel.INFO.value,
                        help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')


def add_learner_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_RANDOM_STATE,
                        type=int,
                        default=1,
                        help='The seed to be used by random number generators. Must be at least 1.')
    parser.add_argument(PARAM_DATA_DIR,
                        type=str,
                        required=True,
                        help='The path of the directory where the data set files are located.')
    parser.add_argument(PARAM_DATASET, type=str, required=True, help='The name of the data set files without suffix.')
    parser.add_argument(PARAM_DATA_SPLIT,
                        type=str,
                        default=DATA_SPLIT_TRAIN_TEST,
                        help='The strategy to be used for splitting the available data into training and test sets. '
                        + 'Must be one of ' + format_dict_keys(DATA_SPLIT_VALUES) + '. For additional options refer to '
                        + 'the documentation.')
    parser.add_argument(PARAM_PRINT_EVALUATION,
                        type=str,
                        default=BooleanOption.TRUE.value,
                        help='Whether the evaluation results should be printed on the console or not. Must be one of '
                        + format_dict_keys(PRINT_EVALUATION_VALUES) + '. For additional options refer to the '
                        + 'documentation.')
    parser.add_argument(PARAM_STORE_EVALUATION,
                        type=str,
                        default=BooleanOption.TRUE.value,
                        help='Whether the evaluation results should be written into output files or not. Must be one '
                        + 'of ' + format_dict_keys(STORE_EVALUATION_VALUES) + '. Does only have an effect if the '
                        + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the '
                        + 'documentation.')
    parser.add_argument(PARAM_EVALUATE_TRAINING_DATA,
                        type=boolean_string,
                        default=False,
                        help='Whether the models should not only be evaluated on the test data, but also on the '
                        + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of binary predictions should be printed on the console or '
                        + 'not. Must be one of ' + format_dict_keys(PRINT_PREDICTION_CHARACTERISTICS_VALUES) + '. Does '
                        + 'only have an effect if the parameter ' + PARAM_PREDICTION_TYPE + ' is set to '
                        + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_STORE_PREDICTION_CHARACTERISTICS,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of binary predictions should be written into output files or '
                        + 'not. Must be one of ' + format_dict_keys(STORE_PREDICTION_CHARACTERISTICS_VALUES) + '. Does '
                        + 'only have an effect if the parameter ' + PARAM_PREDICTION_TYPE + ' is set to '
                        + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PRINT_DATA_CHARACTERISTICS,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of the training data should be printed on the console or '
                        + 'not. Must be one of ' + format_dict_keys(PRINT_DATA_CHARACTERISTICS_VALUES) + '. For '
                        + 'additional options refer to the documentation.')
    parser.add_argument(PARAM_STORE_DATA_CHARACTERISTICS,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the characteristics of the training data should be written into output files or '
                        + 'not. Must be one of ' + format_dict_keys(STORE_DATA_CHARACTERISTICS_VALUES) + '. Does only '
                        + 'have an effect if the parameter ' + PARAM_OUTPUT_DIR + ' is specified. For additional '
                        + 'options refer to the documentation')
    parser.add_argument(PARAM_ONE_HOT_ENCODING,
                        type=boolean_string,
                        default=False,
                        help='Whether one-hot-encoding should be used to encode nominal attributes or not. Must be one '
                        + 'of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_MODEL_DIR, type=str, help='The path of the directory where models should be stored.')
    parser.add_argument(PARAM_PARAMETER_DIR,
                        type=str,
                        help='The path of the directory where configuration files, which specify the parameters to be '
                        + 'used by the algorithm, are located.')
    parser.add_argument(PARAM_OUTPUT_DIR,
                        type=str,
                        help='The path of the directory where experimental results should be saved.')
    parser.add_argument(PARAM_PRINT_PARAMETERS,
                        type=boolean_string,
                        default=False,
                        help='Whether the parameter setting should be printed on the console or not. Must be one of '
                        + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_PARAMETERS,
                        type=boolean_string,
                        default=False,
                        help='Whether the parameter setting should be written into output files or not. Must be one of '
                        + format_enum_values(BooleanOption) + '. Does only have an effect, if the parameter '
                        + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_PREDICTIONS,
                        type=boolean_string,
                        default=False,
                        help='Whether the predictions for individual examples and labels should be printed on the '
                        + 'console or not. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_PREDICTIONS,
                        type=boolean_string,
                        default=False,
                        help='Whether the predictions for individual examples and labels should be written into output '
                        + 'files or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does only have an '
                        + 'effect, if the parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PREDICTION_TYPE,
                        type=str,
                        default=PredictionType.BINARY.value,
                        help='The type of predictions that should be obtained from the learner. Must be one of '
                        + format_enum_values(PredictionType) + '.')


def add_rule_learner_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_INCREMENTAL_EVALUATION,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether models should be evaluated repeatedly, using only a subset of the induced rules '
                        + 'with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                        + 'additional options refer to the documentation.')
    parser.add_argument(PARAM_PRINT_MODEL_CHARACTERISTICS,
                        type=boolean_string,
                        default=False,
                        help='Whether the characteristics of models should be printed on the console or not. Must be '
                        + 'one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_MODEL_CHARACTERISTICS,
                        type=boolean_string,
                        default=False,
                        help='Whether the characteristics of models should be written into output files or not. Must '
                        + 'be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if the '
                        + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_RULES,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the induced rules should be printed on the console or not. Must be one of '
                        + format_dict_keys(PRINT_RULES_VALUES) + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_STORE_RULES,
                        type=str,
                        default=BooleanOption.FALSE.value,
                        help='Whether the induced rules should be written into a text file or not. Must be one of '
                        + format_dict_keys(STORE_RULES_VALUES) + '. Does only have an effect if the parameter '
                        + PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the documentation.')
    parser.add_argument(PARAM_FEATURE_FORMAT,
                        type=str,
                        default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the feature matrix. Must be one of '
                        + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_LABEL_FORMAT,
                        type=str,
                        default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the label matrix. Must be one of '
                        + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_PREDICTION_FORMAT,
                        type=str,
                        default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of predictions. Must be one of '
                        + format_enum_values(SparsePolicy) + '.')


def add_max_rules_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_MAX_RULES,
                        type=int,
                        help='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of '
                        + 'rules should not be restricted.')


def add_time_limit_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_TIME_LIMIT,
                        type=int,
                        help='The duration in seconds after which the induction of rules should be canceled. Must be '
                        + 'at least 1 or 0, if no time limit should be set.')


def add_sequential_post_optimization_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_SEQUENTIAL_POST_OPTIMIZATION,
                        type=str,
                        help='Whether each rule in a previously learned model should be optimized by being relearned '
                        + 'in the context of the other rules. Must be one of ' + format_enum_values(BooleanOption)
                        + '. For additional options refer to the documentation.')


def add_label_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LABEL_SAMPLING,
                        type=str,
                        help='The name of the strategy to be used for label sampling. Must be one of '
                        + format_dict_keys(LABEL_SAMPLING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_instance_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_INSTANCE_SAMPLING,
                        type=str,
                        help='The name of the strategy to be used for instance sampling. Must be one of'
                        + format_dict_keys(INSTANCE_SAMPLING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_feature_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_FEATURE_SAMPLING,
                        type=str,
                        help='The name of the strategy to be used for feature sampling. Must be one of '
                        + format_dict_keys(FEATURE_SAMPLING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_partition_sampling_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARTITION_SAMPLING,
                        type=str,
                        help='The name of the strategy to be used for creating a holdout set. Must be one of '
                        + format_dict_keys(PARTITION_SAMPLING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_feature_binning_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_FEATURE_BINNING,
                        type=str,
                        help='The name of the strategy to be used for feature binning. Must be one of '
                        + format_dict_keys(FEATURE_BINNING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_global_pruning_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_GLOBAL_PRUNING,
                        type=str,
                        help='The name of the strategy to be used for pruning entire rules. Must be one of '
                        + format_dict_keys(GLOBAL_PRUNING_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_rule_pruning_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_RULE_PRUNING,
                        type=str,
                        help='The name of the strategy to be used for pruning individual rules. Must be one of '
                        + format_string_set(RULE_PRUNING_VALUES) + '. Does only have an effect if the parameter '
                        + PARAM_INSTANCE_SAMPLING + ' is not set to "none".')


def add_rule_induction_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_RULE_INDUCTION,
                        type=str,
                        help='The name of the algorithm to be used for the induction of individual rules. Must be one '
                        + 'of ' + format_string_set(RULE_INDUCTION_VALUES) + '. For additional options refer to the '
                        + 'documentation')


def add_parallel_prediction_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_PREDICTION,
                        type=str,
                        help='Whether predictions for different examples should be obtained in parallel or not. Must '
                        + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_parallel_rule_refinement_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT,
                        type=str,
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                        + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to the '
                        + 'documentation.')


def add_parallel_statistic_update_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE,
                        type=str,
                        help='Whether the confusion matrices for different examples should be calculated in parallel '
                        + 'or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options '
                        + 'refer to the documentation')
