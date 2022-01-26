#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
import logging as log
from typing import Dict, Set

from mlrl.boosting.cython.learner import BoostingRuleLearner as BoostingRuleLearnerWrapper, BoostingRuleLearnerConfig
from mlrl.common.cython.learner import RuleLearnerConfig, RuleLearner as RuleLearnerWrapper
from mlrl.common.cython.stopping_criterion import AggregationFunction
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, SAMPLING_WITHOUT_REPLACEMENT, HEAD_TYPE_SINGLE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_NUM_THREADS
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import configure_rule_model_assemblage, configure_rule_induction, \
    configure_feature_binning, configure_label_sampling, configure_instance_sampling, configure_feature_sampling, \
    configure_partition_sampling, configure_pruning, configure_parallel_rule_refinement, \
    configure_parallel_statistic_update, configure_size_stopping_criterion, configure_time_stopping_criterion
from mlrl.common.rule_learners import parse_param, parse_param_and_options, get_num_threads_prediction
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

    def _create_learner(self) -> RuleLearnerWrapper:
        config = BoostingRuleLearnerConfig()
        configure_rule_model_assemblage(config, default_rule=self.default_rule)
        configure_rule_induction(config, min_coverage=int(self.min_coverage), max_conditions=int(self.max_conditions),
                                 max_head_refinements=int(self.max_head_refinements),
                                 recalculate_predictions=self.recalculate_predictions)
        self.__configure_feature_binning(config)
        configure_label_sampling(config, self.feature_sampling)
        configure_instance_sampling(config, self.instance_sampling)
        configure_feature_sampling(config, self.feature_sampling)
        configure_partition_sampling(config, self.holdout)
        configure_pruning(config, self.pruning, self.instance_sampling)
        self.__configure_parallel_rule_refinement(config)
        self.__configure_parallel_statistic_update(config)
        configure_size_stopping_criterion(config, max_rules=self.max_rules)
        configure_time_stopping_criterion(config, time_limit=self.time_limit)
        self.__configure_measure_stopping_criterion(config)
        self.__configure_post_processor(config)
        self.__configure_head_type(config)
        self.__configure_l1_regularization(config)
        self.__configure_l2_regularization(config)
        self.__configure_loss(config)
        self.__configure_label_binning(config)
        self.__configure_classification_predictor(config)
        self.__configure_probability_predictor(config)
        return BoostingRuleLearnerWrapper(config)

    def __configure_feature_binning(self, config: BoostingRuleLearnerConfig):
        feature_binning = self.feature_binning

        if feature_binning == AUTOMATIC:
            config.use_automatic_feature_binning()
        else:
            configure_feature_binning(config, feature_binning)

    def __configure_parallel_rule_refinement(self, config: BoostingRuleLearnerConfig):
        parallel_rule_refinement = self.parallel_rule_refinement

        if parallel_rule_refinement == AUTOMATIC:
            config.use_automatic_parallel_rule_refinement()
        else:
            configure_parallel_rule_refinement(config, parallel_rule_refinement)

    def __configure_parallel_statistic_update(self, config: BoostingRuleLearnerConfig):
        parallel_statistic_update = self.parallel_statistic_update

        if parallel_statistic_update == AUTOMATIC:
            config.use_automatic_parallel_statistic_update()
        else:
            configure_parallel_statistic_update(config, parallel_statistic_update)

    def __configure_measure_stopping_criterion(self, config: RuleLearnerConfig):
        early_stopping = self.early_stopping

        if early_stopping is None:
            config.use_no_measure_stopping_criterion()
        elif self.holdout is None:
            log.warning(
                'Parameter "early_stopping" does not have any effect, because parameter "holdout" is set to "None"!')
        else:
            value, options = parse_param_and_options('early_stopping', early_stopping, EARLY_STOPPING_VALUES)

            if value == EARLY_STOPPING_LOSS:
                aggregation_function = self.__create_aggregation_function(
                    options.get_string(ARGUMENT_AGGREGATION_FUNCTION, 'avg'))
                config.use_measure_stopping_criterion() \
                    .set_aggregation_function(aggregation_function) \
                    .set_min_rules(options.get_int(ARGUMENT_MIN_RULES, 100)) \
                    .set_update_interval(options.get_int(ARGUMENT_UPDATE_INTERVAL, 1)) \
                    .set_stop_interval(options.get_int(ARGUMENT_STOP_INTERVAL, 1)) \
                    .set_num_past(options.get_int(ARGUMENT_NUM_PAST, 50)) \
                    .set_num_current(options.get_int(ARGUMENT_NUM_RECENT, 50)) \
                    .set_min_improvement(options.get_float(ARGUMENT_MIN_IMPROVEMENT, 0.005)) \
                    .set_force_stop(options.get_bool(ARGUMENT_FORCE_STOP, True))

    @staticmethod
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

    def __configure_post_processor(self, config: BoostingRuleLearnerConfig):
        shrinkage = float(self.shrinkage)

        if shrinkage == 1:
            config.use_no_post_processor()
        else:
            config.use_constant_shrinkage_post_processor().set_shrinkage(shrinkage)

    def __configure_head_type(self, config: BoostingRuleLearnerConfig):
        head_type = self.head_type

        if head_type == AUTOMATIC:
            config.use_automatic_heads()
        else:
            value = parse_param("head_type", head_type, HEAD_TYPE_VALUES)

            if value == HEAD_TYPE_SINGLE:
                config.use_single_label_heads()
            elif value == HEAD_TYPE_COMPLETE:
                config.use_complete_heads()

    def __configure_l1_regularization(self, config: BoostingRuleLearnerConfig):
        l1_regularization_weight = float(self.l1_regularization_weight)

        if l1_regularization_weight == 0:
            config.use_no_l1_regularization()
        else:
            config.use_l1_regularization().set_regularization_weight(l1_regularization_weight)

    def __configure_l2_regularization(self, config: BoostingRuleLearnerConfig):
        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight == 0:
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(l2_regularization_weight)

    def __configure_loss(self, config: BoostingRuleLearnerConfig):
        value = parse_param("loss", self.loss, LOSS_VALUES)

        if value == LOSS_SQUARED_ERROR_LABEL_WISE:
            config.use_label_wise_squared_error_loss()
        elif value == LOSS_SQUARED_HINGE_LABEL_WISE:
            config.use_label_wise_squared_hinge_loss()
        elif value == LOSS_LOGISTIC_LABEL_WISE:
            config.use_label_wise_logistic_loss()
        elif value == LOSS_LOGISTIC_EXAMPLE_WISE:
            config.use_example_wise_logistic_loss()

    def __configure_label_binning(self, config: BoostingRuleLearnerConfig):
        label_binning = self.label_binning

        if label_binning is None:
            config.use_no_label_binning()
        elif label_binning == AUTOMATIC:
            config.use_automatic_label_binning()
        else:
            value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

            if value == LABEL_BINNING_EQUAL_WIDTH:
                config.use_equal_width_label_binning() \
                    .set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, 0.04)) \
                    .set_min_bins(options.get_int(ARGUMENT_MIN_BINS, 1)) \
                    .set_max_bins(options.get_int(ARGUMENT_MAX_BINS, 0))

    def __configure_classification_predictor(self, config: BoostingRuleLearnerConfig):
        predictor = self.predictor

        if predictor == AUTOMATIC:
            config.use_automatic_label_binning()
        else:
            value = parse_param('predictor', predictor, PREDICTOR_VALUES)

            if value == PREDICTOR_LABEL_WISE:
                config.use_label_wise_classification_predictor() \
                    .set_num_threads(get_num_threads_prediction(self.parallel_prediction))
            elif value == PREDICTOR_EXAMPLE_WISE:
                config.use_example_wise_classification_predictor() \
                    .set_num_threads(get_num_threads_prediction(self.parallel_prediction))

    def _configure_probability_predictor(self, config: BoostingRuleLearnerConfig):
        config.use_label_wise_probability_predictor() \
            .set_num_threads(get_num_threads_prediction(self.parallel_prediction))
