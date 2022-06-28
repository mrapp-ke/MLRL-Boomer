"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for configuring separate-and-conquer (SeCo) algorithms.
"""
from typing import Dict, Set, Optional

from mlrl.common.options import Options
from mlrl.common.rule_learners import parse_param, parse_param_and_options, NONE
from mlrl.seco.cython.learner import SeCoRuleLearnerConfig

HEAD_TYPE_SINGLE = 'single-label'

HEAD_TYPE_PARTIAL = 'partial'

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

LIFT_FUNCTION_PEAK = 'peak'

ARGUMENT_PEAK_LABEL = 'peak_label'

LIFT_FUNCTION_KLN = 'kln'

ARGUMENT_K = 'k'

ARGUMENT_MAX_LIFT = 'max_lift'

ARGUMENT_CURVATURE = 'curvature'

ARGUMENT_BETA = 'beta'

ARGUMENT_M = 'm'

HEAD_TYPE_VALUES: Set[str] = {
    HEAD_TYPE_SINGLE,
    HEAD_TYPE_PARTIAL
}

LIFT_FUNCTION_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    LIFT_FUNCTION_PEAK: {ARGUMENT_PEAK_LABEL, ARGUMENT_MAX_LIFT, ARGUMENT_CURVATURE},
    LIFT_FUNCTION_KLN: {ARGUMENT_K}
}


def configure_head_type(config: SeCoRuleLearnerConfig, head_type: Optional[str]):
    if head_type is not None:
        value = parse_param('head_type', head_type, HEAD_TYPE_VALUES)

        if value == HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == HEAD_TYPE_PARTIAL:
            config.use_partial_heads()


def configure_lift_function(config: SeCoRuleLearnerConfig, lift_function: Optional[str]):
    if lift_function is not None:
        value, options = parse_param_and_options('lift_function', lift_function, LIFT_FUNCTION_VALUES)

        if value == NONE:
            config.use_no_lift_function()
        elif value == LIFT_FUNCTION_PEAK:
            c = config.use_peak_lift_function()
            c.set_peak_label(options.get_int(ARGUMENT_PEAK_LABEL, c.get_peak_label()))
            c.set_max_lift(options.get_float(ARGUMENT_MAX_LIFT, c.get_max_lift()))
            c.set_curvature(options.get_float(ARGUMENT_CURVATURE, c.get_curvature()))
        elif value == LIFT_FUNCTION_KLN:
            c = config.use_kln_lift_function()
            c.set_k(options.get_float(ARGUMENT_K, c.get_k()))


def configure_precision_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_PRECISION:
        config.use_precision_heuristic()


def configure_precision_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_PRECISION:
        config.use_precision_pruning_heuristic()


def configure_accuracy_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_ACCURACY:
        config.use_accuracy_heuristic()


def configure_accuracy_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_ACCURACY:
        config.use_accuracy_pruning_heuristic()


def configure_recall_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_RECALL:
        config.use_recall_heuristic()


def configure_recall_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_RECALL:
        config.use_recall_pruning_heuristic()


def configure_laplace_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_LAPLACE:
        config.use_laplace_heuristic()


def configure_laplace_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_LAPLACE:
        config.use_laplace_pruning_heuristic()


def configure_wra_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_WRA:
        config.use_wra_heuristic()


def configure_wra_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str):
    if value == HEURISTIC_WRA:
        config.use_wra_pruning_heuristic()


def configure_f_measure_heuristic(config: SeCoRuleLearnerConfig, value: str, options: Options):
    if value == HEURISTIC_F_MEASURE:
        c = config.use_f_measure_heuristic()
        c.set_beta(options.get_float(ARGUMENT_BETA, c.get_beta()))


def configure_f_measure_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str, options: Options):
    if value == HEURISTIC_F_MEASURE:
        c = config.use_f_measure_pruning_heuristic()
        c.set_beta(options.get_float(ARGUMENT_BETA, c.get_beta()))


def configure_m_estimate_heuristic(config: SeCoRuleLearnerConfig, value: str, options: Options):
    if value == HEURISTIC_M_ESTIMATE:
        c = config.use_m_estimate_heuristic()
        c.set_m(options.get_float(ARGUMENT_M, c.get_m()))


def configure_m_estimate_pruning_heuristic(config: SeCoRuleLearnerConfig, value: str, options: Options):
    if value == HEURISTIC_M_ESTIMATE:
        c = config.use_m_estimate_pruning_heuristic()
        c.set_m(options.get_float(ARGUMENT_M, c.get_m()))
