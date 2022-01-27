#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of separate-and-conquer algorithms.
"""
from typing import Dict, Set

from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import PRUNING_IREP, SAMPLING_STRATIFIED_LABEL_WISE
from mlrl.common.rule_learners import configure_rule_model_assemblage, configure_rule_induction, \
    configure_feature_binning, configure_label_sampling, configure_instance_sampling, configure_feature_sampling, \
    configure_partition_sampling, configure_pruning, configure_parallel_rule_refinement, \
    configure_parallel_statistic_update, configure_parallel_prediction, configure_size_stopping_criterion, \
    configure_time_stopping_criterion
from mlrl.common.rule_learners import parse_param, parse_param_and_options
from mlrl.seco.cython.learner import SeCoRuleLearner as SeCoRuleLearnerWrapper, SeCoRuleLearnerConfig
from sklearn.base import ClassifierMixin

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

ARGUMENT_MAX_LIFT = 'max_lift'

ARGUMENT_CURVATURE = 'curvature'

ARGUMENT_BETA = 'beta'

ARGUMENT_M = 'm'

HEAD_TYPE_VALUES: Set[str] = {HEAD_TYPE_SINGLE, HEAD_TYPE_PARTIAL}

HEURISTIC_VALUES: Dict[str, Set[str]] = {
    HEURISTIC_ACCURACY: {},
    HEURISTIC_PRECISION: {},
    HEURISTIC_RECALL: {},
    HEURISTIC_LAPLACE: {},
    HEURISTIC_WRA: {},
    HEURISTIC_F_MEASURE: {ARGUMENT_BETA},
    HEURISTIC_M_ESTIMATE: {ARGUMENT_M}
}

LIFT_FUNCTION_VALUES: Dict[str, Set[str]] = {
    LIFT_FUNCTION_PEAK: {ARGUMENT_PEAK_LABEL, ARGUMENT_MAX_LIFT, ARGUMENT_CURVATURE}
}


class SeCoRuleLearner(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of a Separate-and-Conquer (SeCo) algorithm for learning multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, prediction_format: str = SparsePolicy.AUTO.value,
                 max_rules: int = 500, time_limit: int = 0, head_type: str = HEAD_TYPE_SINGLE,
                 lift_function: str = LIFT_FUNCTION_PEAK, heuristic: str = HEURISTIC_F_MEASURE,
                 pruning_heuristic: str = HEURISTIC_ACCURACY, label_sampling: str = None,
                 instance_sampling: str = SAMPLING_STRATIFIED_LABEL_WISE, feature_sampling: str = None,
                 holdout: str = None, feature_binning: str = None, pruning: str = PRUNING_IREP, min_coverage: int = 1,
                 max_conditions: int = 0, max_head_refinements: int = 1,
                 parallel_rule_refinement: str = BooleanOption.TRUE.value,
                 parallel_statistic_update: str = BooleanOption.FALSE.value,
                 parallel_prediction: str = BooleanOption.TRUE.value):
        """
        :param max_rules:                   The maximum number of rules to be learned (including the default rule). Must
                                            be at least 1 or 0, if the number of rules should not be restricted
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled. Must be at least 1 or 0, if no time limit should be set
        :param head_type:                   The type of the rule heads that should be used. Must be 'single-label' or
                                            'partial'
        :param lift_function:               The lift function that should be used for the induction of partial rule
                                            heads. Must be 'peak'. For additional options refer to the documentation.
        :param heuristic:                   The heuristic to be optimized. Must be 'accuracy', 'precision', 'recall',
                                            'weighted-relative-accuracy', 'f-measure', 'm-estimate' or 'laplace'. For
                                            additional options refer to the documentation
        :param pruning_heuristic:           The heuristic to be optimized when pruning rules. Must be 'accuracy',
                                            'precision', 'recall', 'weighted-relative-accuracy', 'f-measure',
                                            'm-estimate' or 'laplace'. For additional options refer to the documentation
        :param label_sampling:              The strategy that should be used to sample from the available labels
                                            whenever a new rule is learned. Must be 'without-replacement' or `None`, if
                                            no sampling should be used. For additional options refer to the
                                            documentation
        :param instance_sampling:           The strategy that should be used to sample from the available the training
                                            examples whenever a new rule is learned. Must be 'with-replacement',
                                            'without-replacement', 'stratified_label_wise', 'stratified_example_wise' or
                                            `None`, if no sampling should be used. For additional options refer to the
                                            documentation
        :param feature_sampling:            The strategy that is used to sample from the available features whenever a
                                            rule is refined. Must be 'without-replacement' or `None`, if no sampling
                                            should be used. For additional options refer to the documentation
        :param holdout:                     The name of the strategy that should be used to creating a holdout set. Must
                                            be 'random', 'stratified-label-wise', 'stratified-example-wise' or `None`,
                                            if no holdout set should be used. For additional options refer to the
                                            documentation
        :param feature_binning:             The strategy that should be used to assign examples to bins based on their
                                            feature values. Must be 'equal-width', 'equal-frequency' or `None`, if no
                                            feature binning should be used. For additional options refer to the
                                            documentation
        :param pruning:                     The strategy that should be used to prune individual rules. Must be 'irep'
                                            or `None`, if no pruning should be used
        :param min_coverage:                The minimum number of training examples that must be covered by a rule. Must
                                            be at least 1
        :param max_conditions:              The maximum number of conditions to be included in a rule's body. Must be at
                                            least 1 or 0, if the number of conditions should not be restricted
        :param max_head_refinements:        The maximum number of times the head of a rule may be refined after a new
                                            condition has been added to its body. Must be at least 1 or 0, if the number
                                            of refinements should not be restricted
        :param parallel_rule_refinement:    Whether potential refinements of rules should be searched for in parallel or
                                            not. Must be 'true', 'false' or 'auto', if the most suitable strategy should
                                            be chosen automatically depending on the loss function. For additional
                                            options refer to the documentation
        :param parallel_statistic_update:   Whether the confusion matrices for different examples should be updated in
                                            parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                            strategy should be chosen automatically, depending on the loss function. For
                                            additional options refer to the documentation
        :param parallel_prediction:         Whether predictions for different examples should be obtained in parallel or
                                            not. Must be 'true' or 'false'. For additional options refer to the
                                            documentation
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_type = head_type
        self.lift_function = lift_function
        self.heuristic = heuristic
        self.pruning_heuristic = pruning_heuristic
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.pruning = pruning
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        name += '_head-type=' + str(self.head_type)
        name += '_lift-function=' + str(self.lift_function)
        name += '_heuristic=' + str(self.heuristic)
        if self.label_sampling is not None:
            name += '_label-sampling=' + str(self.label_sampling)
        if self.instance_sampling is not None:
            name += '_instance-sampling=' + str(self.instance_sampling)
        if self.feature_sampling is not None:
            name += '_feature-sampling=' + str(self.feature_sampling)
        if self.holdout is not None:
            name += '_holdout=' + str(self.holdout)
        if self.feature_binning is not None:
            name += '_feature-binning=' + str(self.feature_binning)
        if self.pruning is not None:
            name += '_pruning-heuristic=' + str(self.pruning_heuristic)
            name += '_pruning=' + str(self.pruning)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) > 0:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) > 0:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_learner(self) -> RuleLearnerWrapper:
        config = SeCoRuleLearnerConfig()
        configure_rule_model_assemblage(config, default_rule=BooleanOption.TRUE.value)
        configure_rule_induction(config, min_coverage=int(self.min_coverage), max_conditions=int(self.max_conditions),
                                 max_head_refinements=int(self.max_head_refinements),
                                 recalculate_predictions=BooleanOption.FALSE.value)
        configure_feature_binning(config, self.feature_binning)
        configure_label_sampling(config, self.feature_sampling)
        configure_instance_sampling(config, self.instance_sampling)
        configure_feature_sampling(config, self.feature_sampling)
        configure_partition_sampling(config, self.holdout)
        configure_pruning(config, self.pruning, self.instance_sampling)
        configure_parallel_rule_refinement(config, self.parallel_rule_refinement)
        configure_parallel_statistic_update(config, self.parallel_statistic_update)
        configure_parallel_prediction(config, self.parallel_prediction)
        configure_size_stopping_criterion(config, max_rules=self.max_rules)
        configure_time_stopping_criterion(config, time_limit=self.time_limit)
        self.__configure_head_type(config)
        self.__configure_heuristic(config)
        self.__configure_pruning_heuristic(config)
        self.__configure_lift_function(config)
        return SeCoRuleLearnerWrapper(config)

    def __configure_head_type(self, config: SeCoRuleLearnerConfig):
        head_type = parse_param('head_type', self.head_type, HEAD_TYPE_VALUES)

        if head_type == HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif head_type == HEAD_TYPE_PARTIAL:
            config.use_partial_heads()

    def __configure_heuristic(self, config: SeCoRuleLearnerConfig):
        value, options = parse_param_and_options('heuristic', self.heuristic, HEURISTIC_VALUES)

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
            c.set_beta(options.get_float(ARGUMENT_BETA, c.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            c = config.use_m_estimate_heuristic()
            c.set_m(options.get_float(ARGUMENT_M, c.get_m()))

    def __configure_pruning_heuristic(self, config: SeCoRuleLearnerConfig):
        value, options = parse_param_and_options('pruning_heuristic', self.pruning_heuristic, HEURISTIC_VALUES)

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
            c.set_beta(options.get_float(ARGUMENT_BETA, c.get_beta()))
        elif value == HEURISTIC_M_ESTIMATE:
            c = config.use_m_estimate_pruning_heuristic()
            c.set_m(options.get_float(ARGUMENT_M, c.get_m()))

    def __configure_lift_function(self, config: SeCoRuleLearnerConfig):
        value, options = parse_param_and_options('lift_function', self.lift_function, LIFT_FUNCTION_VALUES)

        if value == LIFT_FUNCTION_PEAK:
            c = config.use_peak_lift_function()
            c.set_peak_label(options.get_int(ARGUMENT_PEAK_LABEL, c.get_peak_label()))
            c.set_max_lift(options.get_float(ARGUMENT_MAX_LIFT, c.get_max_lift()))
            c.set_curvature(options.get_float(ARGUMENT_CURVATURE, c.get_curvature()))
