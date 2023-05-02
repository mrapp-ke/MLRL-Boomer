"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments.
"""
from argparse import ArgumentParser

from mlrl.common.config import RULE_INDUCTION_VALUES, LABEL_SAMPLING_VALUES, FEATURE_SAMPLING_VALUES, \
    INSTANCE_SAMPLING_VALUES, PARTITION_SAMPLING_VALUES, FEATURE_BINNING_VALUES, GLOBAL_PRUNING_VALUES, \
    RULE_PRUNING_VALUES, PARALLEL_VALUES
from mlrl.common.format import format_enum_values, format_string_set, format_dict_keys
from mlrl.common.options import BooleanOption

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
