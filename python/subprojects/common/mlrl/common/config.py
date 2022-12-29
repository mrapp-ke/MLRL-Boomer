"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility function for configuring rule learning algorithms.
"""
from mlrl.common.cython.learner import RuleLearnerConfig
from mlrl.common.cython.stopping_criterion import AggregationFunction
from mlrl.common.options import BooleanOption, parse_param, parse_param_and_options
from typing import Dict, Set, Optional

AUTOMATIC = 'auto'

NONE = 'none'

RULE_INDUCTION_TOP_DOWN_GREEDY = 'top-down-greedy'

RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH = 'top-down-beam-search'

ARGUMENT_BEAM_WIDTH = 'beam_width'

ARGUMENT_RESAMPLE_FEATURES = 'resample_features'

ARGUMENT_MIN_COVERAGE = 'min_coverage'

ARGUMENT_MIN_SUPPORT = 'min_support'

ARGUMENT_MAX_CONDITIONS = 'max_conditions'

ARGUMENT_MAX_HEAD_REFINEMENTS = 'max_head_refinements'

ARGUMENT_RECALCULATE_PREDICTIONS = 'recalculate_predictions'

SAMPLING_WITH_REPLACEMENT = 'with-replacement'

SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

SAMPLING_STRATIFIED_LABEL_WISE = 'stratified-label-wise'

SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

ARGUMENT_SAMPLE_SIZE = 'sample_size'

ARGUMENT_NUM_SAMPLES = 'num_samples'

PARTITION_SAMPLING_RANDOM = 'random'

ARGUMENT_HOLDOUT_SET_SIZE = 'holdout_set_size'

EARLY_STOPPING_OBJECTIVE = 'objective'

AGGREGATION_FUNCTION_MIN = 'min'

AGGREGATION_FUNCTION_MAX = 'max'

AGGREGATION_FUNCTION_ARITHMETIC_MEAN = 'avg'

ARGUMENT_NUM_ITERATIONS = 'num_iterations'

ARGUMENT_REFINE_HEADS = 'refine_heads'

ARGUMENT_USE_HOLDOUT_SET = 'use_holdout_set'

ARGUMENT_MIN_RULES = 'min_rules'

ARGUMENT_UPDATE_INTERVAL = 'update_interval'

ARGUMENT_STOP_INTERVAL = 'stop_interval'

ARGUMENT_NUM_PAST = 'num_past'

ARGUMENT_NUM_RECENT = 'num_recent'

ARGUMENT_MIN_IMPROVEMENT = 'min_improvement'

ARGUMENT_FORCE_STOP = 'force_stop'

ARGUMENT_AGGREGATION_FUNCTION = 'aggregation'

BINNING_EQUAL_FREQUENCY = 'equal-frequency'

BINNING_EQUAL_WIDTH = 'equal-width'

ARGUMENT_BIN_RATIO = 'bin_ratio'

ARGUMENT_MIN_BINS = 'min_bins'

ARGUMENT_MAX_BINS = 'max_bins'

RULE_PRUNING_IREP = 'irep'

ARGUMENT_NUM_THREADS = 'num_threads'

RULE_INDUCTION_VALUES: Dict[str, Set[str]] = {
    RULE_INDUCTION_TOP_DOWN_GREEDY: {ARGUMENT_MIN_COVERAGE, ARGUMENT_MIN_SUPPORT, ARGUMENT_MAX_CONDITIONS,
                                     ARGUMENT_MAX_HEAD_REFINEMENTS, ARGUMENT_RECALCULATE_PREDICTIONS},
    RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH: {ARGUMENT_BEAM_WIDTH, ARGUMENT_RESAMPLE_FEATURES, ARGUMENT_MIN_COVERAGE,
                                          ARGUMENT_MIN_SUPPORT, ARGUMENT_MAX_CONDITIONS, ARGUMENT_MAX_HEAD_REFINEMENTS,
                                          ARGUMENT_RECALCULATE_PREDICTIONS}
}

LABEL_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_NUM_SAMPLES}
}

FEATURE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE}
}

INSTANCE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITH_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_SAMPLE_SIZE}
}

PARTITION_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    PARTITION_SAMPLING_RANDOM: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_HOLDOUT_SET_SIZE}
}

SEQUENTIAL_POST_OPTIMIZATION_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_ITERATIONS, ARGUMENT_REFINE_HEADS, ARGUMENT_RESAMPLE_FEATURES},
    str(BooleanOption.FALSE.value): {}
}

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_FREQUENCY: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS}
}

EARLY_STOPPING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    EARLY_STOPPING_OBJECTIVE: {ARGUMENT_AGGREGATION_FUNCTION, ARGUMENT_USE_HOLDOUT_SET, ARGUMENT_MIN_RULES,
                               ARGUMENT_UPDATE_INTERVAL, ARGUMENT_STOP_INTERVAL, ARGUMENT_NUM_PAST, ARGUMENT_NUM_RECENT,
                               ARGUMENT_MIN_IMPROVEMENT, ARGUMENT_FORCE_STOP}
}

RULE_PRUNING_VALUES: Set[str] = {
    NONE,
    RULE_PRUNING_IREP
}

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {}
}


def configure_rule_induction(config: RuleLearnerConfig, rule_induction: Optional[str]):
    if rule_induction is not None:
        value, options = parse_param_and_options('rule_induction', rule_induction, RULE_INDUCTION_VALUES)

        if value == RULE_INDUCTION_TOP_DOWN_GREEDY:
            c = config.use_greedy_top_down_rule_induction()
            c.set_min_coverage(options.get_int(ARGUMENT_MIN_COVERAGE, c.get_min_coverage()))
            c.set_min_support(options.get_float(ARGUMENT_MIN_SUPPORT, c.get_min_support()))
            c.set_max_conditions(options.get_int(ARGUMENT_MAX_CONDITIONS, c.get_max_conditions()))
            c.set_max_head_refinements(options.get_int(ARGUMENT_MAX_HEAD_REFINEMENTS, c.get_max_head_refinements()))
            c.set_recalculate_predictions(options.get_bool(ARGUMENT_RECALCULATE_PREDICTIONS,
                                                           c.are_predictions_recalculated()))
        elif value == RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH:
            c = config.use_beam_search_top_down_rule_induction()
            c.set_beam_width(options.get_int(ARGUMENT_BEAM_WIDTH, c.get_beam_width()))
            c.set_resample_features(options.get_bool(ARGUMENT_RESAMPLE_FEATURES, c.are_features_resampled()))
            c.set_min_coverage(options.get_int(ARGUMENT_MIN_COVERAGE, c.get_min_coverage()))
            c.set_min_support(options.get_float(ARGUMENT_MIN_SUPPORT, c.get_min_support()))
            c.set_max_conditions(options.get_int(ARGUMENT_MAX_CONDITIONS, c.get_max_conditions()))
            c.set_max_head_refinements(options.get_int(ARGUMENT_MAX_HEAD_REFINEMENTS, c.get_max_head_refinements()))
            c.set_recalculate_predictions(options.get_bool(ARGUMENT_RECALCULATE_PREDICTIONS,
                                                           c.are_predictions_recalculated()))


def configure_feature_binning(config: RuleLearnerConfig, feature_binning: Optional[str]):
    if feature_binning is not None:
        value, options = parse_param_and_options('feature_binning', feature_binning, FEATURE_BINNING_VALUES)

        if value == NONE:
            config.use_no_feature_binning()
        elif value == BINNING_EQUAL_FREQUENCY:
            c = config.use_equal_frequency_feature_binning()
            c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))
        elif value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_feature_binning()
            c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))


def configure_label_sampling(config: RuleLearnerConfig, label_sampling: Optional[str]):
    if label_sampling is not None:
        value, options = parse_param_and_options('label_sampling', label_sampling, LABEL_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_label_sampling()
        if value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_label_sampling_without_replacement()
            c.set_num_samples(options.get_int(ARGUMENT_NUM_SAMPLES, c.get_num_samples()))


def configure_instance_sampling(config: RuleLearnerConfig, instance_sampling: Optional[str]):
    if instance_sampling is not None:
        value, options = parse_param_and_options('instance_sampling', instance_sampling, INSTANCE_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_instance_sampling()
        elif value == SAMPLING_WITH_REPLACEMENT:
            c = config.use_instance_sampling_with_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_instance_sampling_without_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))


def configure_feature_sampling(config: RuleLearnerConfig, feature_sampling: Optional[str]):
    if feature_sampling is not None:
        value, options = parse_param_and_options('feature_sampling', feature_sampling, FEATURE_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_feature_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_feature_sampling_without_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))


def configure_partition_sampling(config: RuleLearnerConfig, partition_sampling: Optional[str]):
    if partition_sampling is not None:
        value, options = parse_param_and_options('holdout', partition_sampling, PARTITION_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_partition_sampling()
        elif value == PARTITION_SAMPLING_RANDOM:
            c = config.use_random_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))


def configure_early_stopping_criterion(config: RuleLearnerConfig, early_stopping: Optional[str]):
    if early_stopping is not None:
        value, options = parse_param_and_options('early_stopping', early_stopping, EARLY_STOPPING_VALUES)

        if value == NONE:
            config.use_no_early_stopping_criterion()
        elif value == EARLY_STOPPING_OBJECTIVE:
            c = config.use_early_stopping_criterion()
            aggregation_function = options.get_string(ARGUMENT_AGGREGATION_FUNCTION, None)
            c.set_aggregation_function(__create_aggregation_function(
                aggregation_function) if aggregation_function is not None else c.get_aggregation_function())
            c.set_use_holdout_set(options.get_bool(ARGUMENT_USE_HOLDOUT_SET, c.is_holdout_set_used()))
            c.set_min_rules(options.get_int(ARGUMENT_MIN_RULES, c.get_min_rules()))
            c.set_update_interval(options.get_int(ARGUMENT_UPDATE_INTERVAL, c.get_update_interval()))
            c.set_stop_interval(options.get_int(ARGUMENT_STOP_INTERVAL, c.get_stop_interval()))
            c.set_num_past(options.get_int(ARGUMENT_NUM_PAST, c.get_num_past()))
            c.set_num_current(options.get_int(ARGUMENT_NUM_RECENT, c.get_num_current()))
            c.set_min_improvement(options.get_float(ARGUMENT_MIN_IMPROVEMENT, c.get_min_improvement()))
            c.set_force_stop(options.get_bool(ARGUMENT_FORCE_STOP, c.is_stop_forced()))


def configure_rule_pruning(config: RuleLearnerConfig, rule_pruning: Optional[str]):
    if rule_pruning is not None:
        value = parse_param('rule_pruning', rule_pruning, RULE_PRUNING_VALUES)

        if value == NONE:
            config.use_no_rule_pruning()
        elif value == RULE_PRUNING_IREP:
            config.use_irep_pruning()


def configure_parallel_rule_refinement(config: RuleLearnerConfig, parallel_rule_refinement: Optional[str]):
    if parallel_rule_refinement is not None:
        value, options = parse_param_and_options('parallel_rule_refinement', parallel_rule_refinement, PARALLEL_VALUES)

        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_rule_refinement()
        else:
            c = config.use_parallel_rule_refinement()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))


def configure_parallel_statistic_update(config: RuleLearnerConfig, parallel_statistic_update: Optional[str]):
    if parallel_statistic_update is not None:
        value, options = parse_param_and_options('parallel_statistic_update', parallel_statistic_update,
                                                 PARALLEL_VALUES)

        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_statistic_update()
        else:
            c = config.use_parallel_statistic_update()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))


def configure_parallel_prediction(config: RuleLearnerConfig, parallel_prediction: Optional[str]):
    if parallel_prediction is not None:
        value, options = parse_param_and_options('parallel_prediction', parallel_prediction, PARALLEL_VALUES)

        if value == BooleanOption.TRUE.value:
            c = config.use_parallel_prediction()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))
        else:
            config.use_no_parallel_prediction()


def configure_size_stopping_criterion(config: RuleLearnerConfig, max_rules: Optional[int]):
    if max_rules is not None:
        if max_rules == 0:
            config.use_no_size_stopping_criterion()
        else:
            config.use_size_stopping_criterion().set_max_rules(max_rules)


def configure_time_stopping_criterion(config: RuleLearnerConfig, time_limit: Optional[int]):
    if time_limit is not None:
        if time_limit == 0:
            config.use_no_time_stopping_criterion()
        else:
            config.use_time_stopping_criterion().set_time_limit(time_limit)


def __create_aggregation_function(aggregation_function: str) -> AggregationFunction:
    value = parse_param(ARGUMENT_AGGREGATION_FUNCTION, aggregation_function, {AGGREGATION_FUNCTION_MIN,
                                                                              AGGREGATION_FUNCTION_MAX,
                                                                              AGGREGATION_FUNCTION_ARITHMETIC_MEAN})

    if value == AGGREGATION_FUNCTION_MIN:
        return AggregationFunction.MIN
    elif value == AGGREGATION_FUNCTION_MAX:
        return AggregationFunction.MAX
    elif value == AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
        return AggregationFunction.ARITHMETIC_MEAN


def configure_sequential_post_optimization(config: RuleLearnerConfig, sequential_post_optimization: Optional[str]):
    if sequential_post_optimization is not None:
        value, options = parse_param_and_options('sequential_post_optimization', sequential_post_optimization,
                                                 SEQUENTIAL_POST_OPTIMIZATION_VALUES)

        if value == BooleanOption.FALSE.value:
            config.use_no_sequential_post_optimization()
        elif value == BooleanOption.TRUE.value:
            c = config.use_sequential_post_optimization()
            c.set_num_iterations(options.get_int(ARGUMENT_NUM_ITERATIONS, c.get_num_iterations()))
            c.set_refine_heads(options.get_bool(ARGUMENT_REFINE_HEADS, c.are_heads_refined()))
            c.set_resample_features(options.get_bool(ARGUMENT_RESAMPLE_FEATURES, c.are_features_resampled()))
