"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility function for configuring boosting algorithms.
"""
from mlrl.boosting.cython.learner import BoostingRuleLearnerConfig
from mlrl.common.config import NONE, ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, BINNING_EQUAL_WIDTH
from mlrl.common.options import Options, BooleanOption, parse_param, parse_param_and_options
from typing import Dict, Set, Optional

STATISTIC_FORMAT_DENSE = 'dense'

STATISTIC_FORMAT_SPARSE = 'sparse'

HEAD_TYPE_SINGLE = 'single-label'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

ARGUMENT_LABEL_RATIO = 'label_ratio'

ARGUMENT_MIN_LABELS = 'min_labels'

ARGUMENT_MAX_LABELS = 'max_labels'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

ARGUMENT_THRESHOLD = 'threshold'

ARGUMENT_EXPONENT = 'exponent'

HEAD_TYPE_COMPLETE = 'complete'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

LOSS_SQUARED_HINGE_EXAMPLE_WISE = 'squared-hinge-example-wise'

BINARY_PREDICTOR_LABEL_WISE = 'label-wise'

BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

ARGUMENT_BASED_ON_PROBABILITIES = 'based_on_probabilities'

BINARY_PREDICTOR_GFM = 'gfm'

PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

STATISTIC_FORMAT_VALUES: Set[str] = {STATISTIC_FORMAT_DENSE, STATISTIC_FORMAT_SPARSE}

DEFAULT_RULE_VALUES: Set[str] = {BooleanOption.TRUE.value, BooleanOption.FALSE.value}

HEAD_TYPE_VALUES: Dict[str, Set[str]] = {
    HEAD_TYPE_SINGLE: {},
    HEAD_TYPE_PARTIAL_FIXED: {ARGUMENT_LABEL_RATIO, ARGUMENT_MIN_LABELS, ARGUMENT_MAX_LABELS},
    HEAD_TYPE_PARTIAL_DYNAMIC: {ARGUMENT_THRESHOLD, ARGUMENT_EXPONENT},
    HEAD_TYPE_COMPLETE: {}
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS}
}


def configure_post_processor(config: BoostingRuleLearnerConfig, shrinkage: Optional[float]):
    if shrinkage is not None:
        if shrinkage == 1:
            config.use_no_post_processor()
        else:
            config.use_constant_shrinkage_post_processor().set_shrinkage(shrinkage)


def configure_l1_regularization(config: BoostingRuleLearnerConfig, l1_regularization_weight: Optional[float]):
    if l1_regularization_weight is not None:
        if l1_regularization_weight == 0:
            config.use_no_l1_regularization()
        else:
            config.use_l1_regularization().set_regularization_weight(l1_regularization_weight)


def configure_l2_regularization(config: BoostingRuleLearnerConfig, l2_regularization_weight: Optional[float]):
    if l2_regularization_weight is not None:
        if l2_regularization_weight == 0:
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(l2_regularization_weight)


def configure_default_rule(config: BoostingRuleLearnerConfig, default_rule: Optional[str]):
    if default_rule is not None:
        value = parse_param('default_rule', default_rule, DEFAULT_RULE_VALUES)

        if value == BooleanOption.TRUE.value:
            config.use_default_rule()
        else:
            config.use_no_default_rule()


def configure_head_type(config: BoostingRuleLearnerConfig, head_type: Optional[str]):
    if head_type is not None:
        value, options = parse_param_and_options("head_type", head_type, HEAD_TYPE_VALUES)

        if value == HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == HEAD_TYPE_PARTIAL_FIXED:
            c = config.use_fixed_partial_heads()
            c.set_label_ratio(options.get_float(ARGUMENT_LABEL_RATIO, c.get_label_ratio()))
            c.set_min_labels(options.get_int(ARGUMENT_MIN_LABELS, c.get_min_labels()))
            c.set_max_labels(options.get_int(ARGUMENT_MAX_LABELS, c.get_max_labels()))
        elif value == HEAD_TYPE_PARTIAL_DYNAMIC:
            c = config.use_dynamic_partial_heads()
            c.set_threshold(options.get_float(ARGUMENT_THRESHOLD, c.get_threshold()))
            c.set_exponent(options.get_float(ARGUMENT_EXPONENT, c.get_exponent()))
        elif value == HEAD_TYPE_COMPLETE:
            config.use_complete_heads()


def configure_statistics(config: BoostingRuleLearnerConfig, statistic_format: Optional[str]):
    if statistic_format is not None:
        value = parse_param("statistic_format", statistic_format, STATISTIC_FORMAT_VALUES)

        if value == STATISTIC_FORMAT_DENSE:
            config.use_dense_statistics()
        elif value == STATISTIC_FORMAT_SPARSE:
            config.use_sparse_statistics()


def configure_label_binning(config: BoostingRuleLearnerConfig, label_binning: Optional[str]):
    if label_binning is not None:
        value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

        if value == NONE:
            config.use_no_label_binning()
        if value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_label_binning()
            c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))


def configure_label_wise_squared_error_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_SQUARED_ERROR_LABEL_WISE:
        config.use_label_wise_squared_error_loss()


def configure_label_wise_squared_hinge_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_SQUARED_HINGE_LABEL_WISE:
        config.use_label_wise_squared_hinge_loss()


def configure_label_wise_logistic_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_LOGISTIC_LABEL_WISE:
        config.use_label_wise_logistic_loss()


def configure_example_wise_logistic_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_LOGISTIC_EXAMPLE_WISE:
        config.use_example_wise_logistic_loss()


def configure_example_wise_squared_error_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_SQUARED_ERROR_EXAMPLE_WISE:
        config.use_example_wise_squared_error_loss()


def configure_example_wise_squared_hinge_loss(config: BoostingRuleLearnerConfig, value: str):
    if value == LOSS_SQUARED_HINGE_EXAMPLE_WISE:
        config.use_example_wise_squared_hinge_loss()


def configure_label_wise_binary_predictor(config: BoostingRuleLearnerConfig, value: str, options: Options):
    if value == BINARY_PREDICTOR_LABEL_WISE:
        c = config.use_label_wise_binary_predictor()
        c.set_based_on_probabilities(options.get_bool(ARGUMENT_BASED_ON_PROBABILITIES, c.is_based_on_probabilities()))


def configure_example_wise_binary_predictor(config: BoostingRuleLearnerConfig, value: str, options: Options):
    if value == BINARY_PREDICTOR_EXAMPLE_WISE:
        c = config.use_example_wise_binary_predictor()
        c.set_based_on_probabilities(options.get_bool(ARGUMENT_BASED_ON_PROBABILITIES, c.is_based_on_probabilities()))


def configure_gfm_binary_predictor(config: BoostingRuleLearnerConfig, value: str, options: Options):
    if value == BINARY_PREDICTOR_GFM:
        c = config.use_gfm_binary_predictor()
        c.set_based_on_probabilities(options.get_bool(ARGUMENT_BASED_ON_PROBABILITIES, c.is_based_on_probabilities()))


def configure_label_wise_probability_predictor(config: BoostingRuleLearnerConfig, value: str):
    if value == PROBABILITY_PREDICTOR_LABEL_WISE:
        config.use_label_wise_probability_predictor()


def configure_marginalized_probability_predictor(config: BoostingRuleLearnerConfig, value: str):
    if value == PROBABILITY_PREDICTOR_MARGINALIZED:
        config.use_marginalized_probability_predictor()
