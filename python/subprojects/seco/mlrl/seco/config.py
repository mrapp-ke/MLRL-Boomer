"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities that ease the configuration of separate-and-conquer (SeCo) algorithms.
"""
from mlrl.seco.cython.learner import SingleLabelHeadMixin, PartialHeadMixin, NoLiftFunctionMixin, \
    PeakLiftFunctionMixin, KlnLiftFunctionMixin, AccuracyHeuristicMixin, PrecisionHeuristicMixin, \
    RecallHeuristicMixin, LaplaceHeuristicMixin, WraHeuristicMixin, FMeasureHeuristicMixin, MEstimateHeuristicMixin, \
    AccuracyPruningHeuristicMixin, PrecisionPruningHeuristicMixin, RecallPruningHeuristicMixin, \
    LaplacePruningHeuristicMixin, WraPruningHeuristicMixin, FMeasurePruningHeuristicMixin, \
    MEstimatePruningHeuristicMixin
from mlrl.common.config import NominalParameter, NONE, RULE_LEARNER_PARAMETERS
from mlrl.common.options import Options, parse_param, parse_param_and_options
from typing import Dict, Set, Optional

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

OPTION_PEAK_LABEL = 'peak_label'

LIFT_FUNCTION_KLN = 'kln'

OPTION_K = 'k'

OPTION_MAX_LIFT = 'max_lift'

OPTION_CURVATURE = 'curvature'

OPTION_BETA = 'beta'

OPTION_M = 'm'

HEAD_TYPE_VALUES: Set[str] = {HEAD_TYPE_SINGLE, HEAD_TYPE_PARTIAL}

LIFT_FUNCTION_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    LIFT_FUNCTION_PEAK: {OPTION_PEAK_LABEL, OPTION_MAX_LIFT, OPTION_CURVATURE},
    LIFT_FUNCTION_KLN: {OPTION_K}
}


def configure_head_type(config, head_type: Optional[str]):
    if head_type is not None:
        value = parse_param('head_type', head_type, HEAD_TYPE_VALUES)

        if value == HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == HEAD_TYPE_PARTIAL:
            config.use_partial_heads()


def configure_lift_function(config, lift_function: Optional[str]):
    if lift_function is not None:
        value, options = parse_param_and_options('lift_function', lift_function, LIFT_FUNCTION_VALUES)

        if value == NONE:
            config.use_no_lift_function()
        elif value == LIFT_FUNCTION_PEAK:
            c = config.use_peak_lift_function()
            c.set_peak_label(options.get_int(OPTION_PEAK_LABEL, c.get_peak_label()))
            c.set_max_lift(options.get_float(OPTION_MAX_LIFT, c.get_max_lift()))
            c.set_curvature(options.get_float(OPTION_CURVATURE, c.get_curvature()))
        elif value == LIFT_FUNCTION_KLN:
            c = config.use_kln_lift_function()
            c.set_k(options.get_float(OPTION_K, c.get_k()))


def configure_precision_heuristic(config, value: str):
    if value == HEURISTIC_PRECISION:
        config.use_precision_heuristic()


def configure_precision_pruning_heuristic(config, value: str):
    if value == HEURISTIC_PRECISION:
        config.use_precision_pruning_heuristic()


def configure_accuracy_heuristic(config, value: str):
    if value == HEURISTIC_ACCURACY:
        config.use_accuracy_heuristic()


def configure_accuracy_pruning_heuristic(config, value: str):
    if value == HEURISTIC_ACCURACY:
        config.use_accuracy_pruning_heuristic()


def configure_recall_heuristic(config, value: str):
    if value == HEURISTIC_RECALL:
        config.use_recall_heuristic()


def configure_recall_pruning_heuristic(config, value: str):
    if value == HEURISTIC_RECALL:
        config.use_recall_pruning_heuristic()


def configure_laplace_heuristic(config, value: str):
    if value == HEURISTIC_LAPLACE:
        config.use_laplace_heuristic()


def configure_laplace_pruning_heuristic(config, value: str):
    if value == HEURISTIC_LAPLACE:
        config.use_laplace_pruning_heuristic()


def configure_wra_heuristic(config, value: str):
    if value == HEURISTIC_WRA:
        config.use_wra_heuristic()


def configure_wra_pruning_heuristic(config, value: str):
    if value == HEURISTIC_WRA:
        config.use_wra_pruning_heuristic()


def configure_f_measure_heuristic(config, value: str, options: Options):
    if value == HEURISTIC_F_MEASURE:
        c = config.use_f_measure_heuristic()
        c.set_beta(options.get_float(OPTION_BETA, c.get_beta()))


def configure_f_measure_pruning_heuristic(config, value: str, options: Options):
    if value == HEURISTIC_F_MEASURE:
        c = config.use_f_measure_pruning_heuristic()
        c.set_beta(options.get_float(OPTION_BETA, c.get_beta()))


def configure_m_estimate_heuristic(config, value: str, options: Options):
    if value == HEURISTIC_M_ESTIMATE:
        c = config.use_m_estimate_heuristic()
        c.set_m(options.get_float(OPTION_M, c.get_m()))


def configure_m_estimate_pruning_heuristic(config, value: str, options: Options):
    if value == HEURISTIC_M_ESTIMATE:
        c = config.use_m_estimate_pruning_heuristic()
        c.set_m(options.get_float(OPTION_M, c.get_m()))


class HeadTypeParameter(NominalParameter):
    """
    A parameter that allows to configure the type of the rule heads that should be used.
    """

    HEAD_TYPE_SINGLE = 'single-label'

    HEAD_TYPE_PARTIAL = 'partial'

    def __init__(self):
        super().__init__(name='head_type', description='The type of the rule heads that should be used')
        self.add_value(name=self.HEAD_TYPE_SINGLE, mixin=SingleLabelHeadMixin)
        self.add_value(name=self.HEAD_TYPE_PARTIAL, mixin=PartialHeadMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == self.HEAD_TYPE_PARTIAL:
            config.use_partial_heads()


class LiftFunctionParameter(NominalParameter):
    """
    A parameter that allows to configure the lift function to be used for the induction of multi-label rules.
    """

    LIFT_FUNCTION_PEAK = 'peak'

    OPTION_PEAK_LABEL = 'peak_label'

    OPTION_MAX_LIFT = 'max_lift'

    OPTION_CURVATURE = 'curvature'

    LIFT_FUNCTION_KLN = 'kln'

    OPTION_K = 'k'

    def __init__(self):
        super().__init__(name='lift_function',
                         description='The lift function to be used for the induction of multi-label rules')
        self.add_value(name=NONE, mixin=NoLiftFunctionMixin)
        self.add_value(name=self.LIFT_FUNCTION_PEAK,
                       mixin=PeakLiftFunctionMixin,
                       options={self.OPTION_PEAK_LABEL, self.OPTION_MAX_LIFT, self.OPTION_CURVATURE})
        self.add_value(name=self.LIFT_FUNCTION_KLN, mixin=KlnLiftFunctionMixin, options={self.OPTION_K})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_lift_function()
        elif value == self.LIFT_FUNCTION_PEAK:
            c = config.use_peak_lift_function()
            c.set_peak_label(options.get_int(self.OPTION_PEAK_LABEL, c.get_peak_label()))
            c.set_max_lift(options.get_float(self.OPTION_MAX_LIFT, c.get_max_lift()))
            c.set_curvature(options.get_float(self.OPTION_CURVATURE, c.get_curvature()))
        elif value == self.LIFT_FUNCTION_KLN:
            c = config.use_kln_lift_function()
            c.set_k(options.get_float(self.OPTION_K, c.get_k()))        


class HeuristicParameter(NominalParameter):
    """
    A parameter that allows to configure the heuristic to be used for learning rules.
    """

    def __init__(self):
        super().__init__(name='heuristic', description='The name of the heuristic to be used for learning rules')
        self.add_value(name=HEURISTIC_ACCURACY, mixin=AccuracyHeuristicMixin)
        self.add_value(name=HEURISTIC_PRECISION, mixin=PrecisionHeuristicMixin)
        self.add_value(name=HEURISTIC_RECALL, mixin=RecallHeuristicMixin)
        self.add_value(name=HEURISTIC_LAPLACE, mixin=LaplaceHeuristicMixin)
        self.add_value(name=HEURISTIC_WRA, mixin=WraHeuristicMixin)
        self.add_value(name=HEURISTIC_F_MEASURE, mixin=FMeasureHeuristicMixin, options={OPTION_BETA})
        self.add_value(name=HEURISTIC_M_ESTIMATE, mixin=MEstimateHeuristicMixin, options={OPTION_M})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == HEURISTIC_ACCURACY:
            config.use_accuracy_heuristic()
        elif value == HEURISTIC_PRECISION:
            config.use_precision_heuristic()
        elif value == HEURISTIC_RECALL:
            config.use_recall_heuristic()
        elif value == HEURISTIC_LAPLACE:
            config.use_laplace_heuristic()
        elif value == HEURISTIC_WRA:
            config.use_wra_heuristic()
        elif value == HEURISTIC_F_MEASURE:
            c = config.use_f_measure_heuristic()
            c.set_beta(options.get_float(OPTION_BETA, c.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            c = config.use_m_estimate_heuristic()
            c.set_m(options.get_float(OPTION_M, c.get_m()))


class PruningHeuristicParameter(NominalParameter):
    """
    A parameter that allows to configure the heuristic to be used for pruning individual rules.
    """

    def __init__(self):
        super().__init__(name='pruning_heuristic',
                         description='The name of the heuristic to be used for pruning individual rules')
        self.add_value(name=HEURISTIC_ACCURACY, mixin=AccuracyPruningHeuristicMixin)
        self.add_value(name=HEURISTIC_PRECISION, mixin=PrecisionPruningHeuristicMixin)
        self.add_value(name=HEURISTIC_RECALL, mixin=RecallPruningHeuristicMixin)
        self.add_value(name=HEURISTIC_LAPLACE, mixin=LaplacePruningHeuristicMixin)
        self.add_value(name=HEURISTIC_WRA, mixin=WraPruningHeuristicMixin)
        self.add_value(name=HEURISTIC_F_MEASURE, mixin=FMeasurePruningHeuristicMixin, options={OPTION_BETA})
        self.add_value(name=HEURISTIC_M_ESTIMATE, mixin=MEstimatePruningHeuristicMixin, options={OPTION_M})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == HEURISTIC_ACCURACY:
            config.use_accuracy_pruning_heuristic()
        elif value == HEURISTIC_PRECISION:
            config.use_precision_pruning_heuristic()
        elif value == HEURISTIC_RECALL:
            config.use_recall_pruning_heuristic()
        elif value == HEURISTIC_LAPLACE:
            config.use_laplace_pruning_heuristic()
        elif value == HEURISTIC_WRA:
            config.use_wra_pruning_heuristic()
        elif value == HEURISTIC_F_MEASURE:
            c = config.use_f_measure_pruning_heuristic()
            c.set_beta(options.get_float(OPTION_BETA, c.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            c = config.use_m_estimate_pruning_heuristic()
            c.set_m(options.get_float(OPTION_M, c.get_m()))


SECO_RULE_LEARNER_PARAMETERS = RULE_LEARNER_PARAMETERS + [
    HeadTypeParameter(),
    LiftFunctionParameter(),
    HeuristicParameter(),
    PruningHeuristicParameter()
]
