#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
from typing import Dict, Set

from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, SAMPLING_WITHOUT_REPLACEMENT, HEAD_TYPE_SINGLE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_NUM_THREADS
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from sklearn.base import ClassifierMixin

EARLY_STOPPING_LOSS = 'loss'

AGGREGATION_FUNCTION_MIN = 'min'

AGGREGATION_FUNCTION_MAX = 'max'

AGGREGATION_FUNCTION_ARITHMETIC_MEAN = 'avg'

ARGUMENT_MIN_RULES = 'min_rules'

ARGUMENT_UPDATE_INTERVAL = 'update_interval'

ARGUMENT_STOP_INTERVAL = 'stop_interval'

ARGUMENT_NUM_PAST = 'num_past'

ARGUMENT_NUM_RECENT = 'num_recent'

ARGUMENT_MIN_IMPROVEMENT = 'min_improvement'

ARGUMENT_FORCE_STOP = 'force_stop'

ARGUMENT_AGGREGATION_FUNCTION = 'aggregation'

HEAD_TYPE_COMPLETE = 'complete'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

NON_DECOMPOSABLE_LOSSES = {LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_LABEL_WISE = 'label-wise'

PREDICTOR_EXAMPLE_WISE = 'example-wise'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

HEAD_TYPE_VALUES: Set[str] = {HEAD_TYPE_SINGLE, HEAD_TYPE_COMPLETE, AUTOMATIC}

EARLY_STOPPING_VALUES: Dict[str, Set[str]] = {
    EARLY_STOPPING_LOSS: {ARGUMENT_AGGREGATION_FUNCTION, ARGUMENT_MIN_RULES, ARGUMENT_UPDATE_INTERVAL,
                          ARGUMENT_STOP_INTERVAL, ARGUMENT_NUM_PAST, ARGUMENT_NUM_RECENT, ARGUMENT_MIN_IMPROVEMENT,
                          ARGUMENT_FORCE_STOP}
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    LABEL_BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {}
}

LOSS_VALUES: Set[str] = {LOSS_SQUARED_ERROR_LABEL_WISE, LOSS_SQUARED_HINGE_LABEL_WISE, LOSS_LOGISTIC_LABEL_WISE,
                         LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_VALUES: Set[str] = {PREDICTOR_LABEL_WISE, PREDICTOR_EXAMPLE_WISE, AUTOMATIC}

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {},
    AUTOMATIC: {}
}


class Boomer(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of "BOOMER", an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, prediction_format: str = SparsePolicy.AUTO.value,
                 max_rules: int = 1000, default_rule: str = BooleanOption.TRUE.value, time_limit: int = 0,
                 early_stopping: str = None, head_type: str = AUTOMATIC, loss: str = LOSS_LOGISTIC_LABEL_WISE,
                 predictor: str = AUTOMATIC, label_sampling: str = None, instance_sampling: str = None,
                 recalculate_predictions: str = BooleanOption.TRUE.value,
                 feature_sampling: str = SAMPLING_WITHOUT_REPLACEMENT, holdout: str = None, feature_binning: str = None,
                 label_binning: str = AUTOMATIC, pruning: str = None, shrinkage: float = 0.3,
                 l1_regularization_weight: float = 0.0, l2_regularization_weight: float = 1.0, min_coverage: int = 1,
                 max_conditions: int = 0, max_head_refinements: int = 1, parallel_rule_refinement: str = AUTOMATIC,
                 parallel_statistic_update: str = AUTOMATIC, parallel_prediction: str = BooleanOption.TRUE.value):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param default_rule:                        Whether a default rule should be used, or not. Must be `true` or
                                                    `false`
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param early_stopping:                      The strategy that is used for early stopping. Must be `measure` or
                                                    None, if no early stopping should be used
        :param head_type:                           The type of the rule heads that should be used. Must be
                                                    `single-label`, `complete` or 'auto', if the type of the heads
                                                    should be chosen automatically
        :param loss:                                The loss function to be minimized. Must be
                                                    `squared-error-label-wise`, `logistic-label-wise` or
                                                    `logistic-example-wise`
        :param predictor:                           The strategy that is used for making predictions. Must be
                                                    `label-wise`, `example-wise` or `auto`, if the most suitable
                                                    strategy should be chosen automatically depending on the loss
                                                    function
        :param label_sampling:                      The strategy that is used for sampling the labels each time a new
                                                    classification rule is learned. Must be 'without-replacement' or
                                                    None, if no sampling should be used. Additional options may be
                                                    provided using the bracket notation
                                                    `without-replacement{num_samples=5}`
        :param instance_sampling:                   The strategy that is used for sampling the training examples each
                                                    time a new classification rule is learned. Must be
                                                    `with-replacement`, `without-replacement`, `stratified_label_wise`,
                                                    `stratified_example_wise` or None, if no sampling should be used.
                                                    Additional options may be provided using the bracket notation
                                                    `with-replacement{sample_size=0.5}`
        :param recalculate_predictions:             Whether the predictions of rules should be recalculated on the
                                                    entire training data, if instance sampling is used, or not. Must be
                                                    `true` or `false`
        :param feature_sampling:                    The strategy that is used for sampling the features each time a
                                                    classification rule is refined. Must be `without-replacement` or
                                                    None, if no sampling should be used. Additional options may be
                                                    provided using the bracket notation
                                                    `without-replacement{sample_size=0.5}`
        :param holdout:                             The name of the strategy to be used for creating a holdout set. Must
                                                    be `random` or None, if no holdout set should be used. Additional
                                                    options may be provided using the bracket notation
                                                    `random{holdout_set_size=0.5}`
        :param feature_binning:                     The strategy that is used for assigning examples to bins based on
                                                    their feature values. Must be `equal-width`, `equal-frequency` or
                                                    None, if no feature binning should be used. Additional options may
                                                    be provided using the bracket notation `equal-width{bin_ratio=0.5}`
        :param label_binning:                       The strategy that is used for assigning labels to bins. Must be
                                                    `auto`, `equal-width` or None, if no label binning should be used.
                                                    Additional options may be provided using the bracket notation
                                                    `equal-width{bin_ratio=0.04,min_bins=1,max_bins=0}`. If `auto` is
                                                    used, the most suitable strategy is chosen automatically based on
                                                    the loss function and the type of rule heads
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param shrinkage:                           The shrinkage parameter that should be applied to the predictions of
                                                    newly induced rules to reduce their effect on the entire model. Must
                                                    be in (0, 1]
        :param l1_regularization_weight:            The weight of the L1 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param l2_regularization_weight:            The weight of the L2 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or 0, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    0, if the number of refinements should not be restricted
        :param parallel_rule_refinement:            Whether potential refinements of rules should be searched for in
                                                    parallel or not. Must be `true`, `false` or `auto`, if the most
                                                    suitable strategy should be chosen automatically depending on the
                                                    loss function. Additional options may be provided using the bracket
                                                    notation `true{num_threads=8}`
        :param parallel_statistic_update:           Whether the gradients and Hessians for different examples should be
                                                    calculated in parallel or not. Must be `true`, `false` or `auto`, if
                                                    the most suitable strategy should be chosen automatically depending
                                                    on the loss function. Additional options may be provided using the
                                                    bracket notation `true{num_threads=8}`
        :param parallel_prediction:                 Whether predictions for different examples should be obtained in
                                                    parallel or not. Must be `true` or `false`. Additional options may
                                                    be provided using the bracket notation `true{num_threads=8}`
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.max_rules = max_rules
        self.default_rule = default_rule
        self.time_limit = time_limit
        self.early_stopping = early_stopping
        self.head_type = head_type
        self.loss = loss
        self.predictor = predictor
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.recalculate_predictions = recalculate_predictions
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.early_stopping is not None:
            name += '_early-stopping=' + str(self.early_stopping)
        if self.head_type != AUTOMATIC:
            name += '_head-type=' + str(self.head_type)
        name += '_loss=' + str(self.loss)
        if self.predictor != AUTOMATIC:
            name += '_predictor=' + str(self.predictor)
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
        if self.label_binning is not None and self.label_binning != AUTOMATIC:
            name += '_label-binning=' + str(self.label_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l1_regularization_weight) > 0.0:
            name += '_l1=' + str(self.l1_regularization_weight)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) > 0:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) > 0:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name
