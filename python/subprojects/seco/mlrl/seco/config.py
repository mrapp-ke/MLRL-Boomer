"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities that ease the configuration of separate-and-conquer (SeCo) algorithms.
"""
from typing import Optional

from mlrl.common.config import NONE, RULE_LEARNER_PARAMETERS, FeatureBinningParameter, NominalParameter
from mlrl.common.options import Options

from mlrl.seco.cython.learner import AccuracyHeuristicMixin, AccuracyPruningHeuristicMixin, FMeasureHeuristicMixin, \
    FMeasurePruningHeuristicMixin, KlnLiftFunctionMixin, LaplaceHeuristicMixin, LaplacePruningHeuristicMixin, \
    MEstimateHeuristicMixin, MEstimatePruningHeuristicMixin, NoLiftFunctionMixin, PartialHeadMixin, \
    PeakLiftFunctionMixin, PrecisionHeuristicMixin, PrecisionPruningHeuristicMixin, RecallHeuristicMixin, \
    RecallPruningHeuristicMixin, SingleOutputHeadMixin, WraHeuristicMixin, WraPruningHeuristicMixin

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_RECALL = 'recall'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

OPTION_BETA = 'beta'

HEURISTIC_M_ESTIMATE = 'm-estimate'

OPTION_M = 'm'


class HeadTypeParameter(NominalParameter):
    """
    A parameter that allows to configure the type of the rule heads that should be used.
    """

    HEAD_TYPE_SINGLE = 'single'

    HEAD_TYPE_PARTIAL = 'partial'

    def __init__(self):
        super().__init__(name='head_type', description='The type of the rule heads that should be used')
        self.add_value(name=self.HEAD_TYPE_SINGLE, mixin=SingleOutputHeadMixin)
        self.add_value(name=self.HEAD_TYPE_PARTIAL, mixin=PartialHeadMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.HEAD_TYPE_SINGLE:
            config.use_single_output_heads()
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
            conf = config.use_peak_lift_function()
            conf.set_peak_label(options.get_int(self.OPTION_PEAK_LABEL, conf.get_peak_label()))
            conf.set_max_lift(options.get_float(self.OPTION_MAX_LIFT, conf.get_max_lift()))
            conf.set_curvature(options.get_float(self.OPTION_CURVATURE, conf.get_curvature()))
        elif value == self.LIFT_FUNCTION_KLN:
            conf = config.use_kln_lift_function()
            conf.set_k(options.get_float(self.OPTION_K, conf.get_k()))


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
            conf = config.use_f_measure_heuristic()
            conf.set_beta(options.get_float(OPTION_BETA, conf.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            conf = config.use_m_estimate_heuristic()
            conf.set_m(options.get_float(OPTION_M, conf.get_m()))


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
            conf = config.use_f_measure_pruning_heuristic()
            conf.set_beta(options.get_float(OPTION_BETA, conf.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            conf = config.use_m_estimate_pruning_heuristic()
            conf.set_m(options.get_float(OPTION_M, conf.get_m()))


SECO_CLASSIFIER_PARAMETERS = RULE_LEARNER_PARAMETERS | {
    FeatureBinningParameter(),
    HeadTypeParameter(),
    LiftFunctionParameter(),
    HeuristicParameter(),
    PruningHeuristicParameter()
}
