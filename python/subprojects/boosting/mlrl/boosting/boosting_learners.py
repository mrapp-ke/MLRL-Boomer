"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
import mlrl.common.config as common_config
from mlrl.boosting.cython.learner_boomer import Boomer as BoomerWrapper, BoomerConfig
from mlrl.common.config import AUTOMATIC
from mlrl.common.config import configure_rule_induction, configure_feature_binning, configure_label_sampling, \
    configure_instance_sampling, configure_feature_sampling, configure_partition_sampling, configure_rule_pruning, \
    configure_parallel_rule_refinement, configure_parallel_statistic_update, configure_parallel_prediction, \
    configure_size_stopping_criterion, configure_time_stopping_criterion, configure_global_pruning, \
    configure_sequential_post_optimization
from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.options import parse_param, parse_param_and_options
from mlrl.common.rule_learners import RuleLearner, SparsePolicy, get_string, get_int, get_float
from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from typing import Dict, Set, Optional

import mlrl.boosting.config as boosting_config
from mlrl.boosting.config import LOSS_SQUARED_ERROR_LABEL_WISE, LOSS_SQUARED_HINGE_LABEL_WISE, \
    LOSS_LOGISTIC_LABEL_WISE, LOSS_SQUARED_ERROR_EXAMPLE_WISE, LOSS_SQUARED_HINGE_EXAMPLE_WISE, \
    LOSS_LOGISTIC_EXAMPLE_WISE, BINARY_PREDICTOR_LABEL_WISE, BINARY_PREDICTOR_EXAMPLE_WISE, BINARY_PREDICTOR_GFM, \
    PROBABILITY_PREDICTOR_LABEL_WISE, PROBABILITY_PREDICTOR_MARGINALIZED, ARGUMENT_BASED_ON_PROBABILITIES
from mlrl.boosting.config import configure_post_processor, configure_l1_regularization, configure_l2_regularization, \
    configure_default_rule, configure_head_type, configure_statistics, configure_label_wise_squared_error_loss, \
    configure_label_wise_squared_hinge_loss, configure_label_wise_logistic_loss, configure_example_wise_logistic_loss, \
    configure_example_wise_squared_error_loss, configure_example_wise_squared_hinge_loss, configure_label_binning, \
    configure_label_wise_binary_predictor, configure_example_wise_binary_predictor, configure_gfm_binary_predictor, \
    configure_label_wise_probability_predictor, configure_marginalized_probability_predictor

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {**common_config.FEATURE_BINNING_VALUES, **{AUTOMATIC: {}}}

PARALLEL_VALUES: Dict[str, Set[str]] = {**common_config.PARALLEL_VALUES, **{AUTOMATIC: {}}}

STATISTIC_FORMAT_VALUES: Set[str] = boosting_config.STATISTIC_FORMAT_VALUES.union(AUTOMATIC)

DEFAULT_RULE_VALUES: Set[str] = boosting_config.DEFAULT_RULE_VALUES.union(AUTOMATIC)

PARTITION_SAMPLING_VALUES: Dict[str, Set[str]] = {**common_config.PARTITION_SAMPLING_VALUES, **{AUTOMATIC: {}}}

HEAD_TYPE_VALUES: Dict[str, Set[str]] = {**boosting_config.HEAD_TYPE_VALUES, **{AUTOMATIC: {}}}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {**boosting_config.LABEL_BINNING_VALUES, **{AUTOMATIC: {}}}

LOSS_VALUES: Set[str] = {
    LOSS_SQUARED_ERROR_LABEL_WISE, LOSS_SQUARED_ERROR_EXAMPLE_WISE, LOSS_SQUARED_HINGE_LABEL_WISE,
    LOSS_SQUARED_HINGE_EXAMPLE_WISE, LOSS_LOGISTIC_LABEL_WISE, LOSS_LOGISTIC_EXAMPLE_WISE
}

BINARY_PREDICTOR_VALUES: Dict[str, Set[str]] = {
    BINARY_PREDICTOR_LABEL_WISE: {ARGUMENT_BASED_ON_PROBABILITIES},
    BINARY_PREDICTOR_EXAMPLE_WISE: {ARGUMENT_BASED_ON_PROBABILITIES},
    BINARY_PREDICTOR_GFM: {ARGUMENT_BASED_ON_PROBABILITIES},
    AUTOMATIC: {}
}

PROBABILITY_PREDICTOR_VALUES: Set[str] = {
    PROBABILITY_PREDICTOR_LABEL_WISE, PROBABILITY_PREDICTOR_MARGINALIZED, AUTOMATIC
}


class Boomer(RuleLearner, ClassifierMixin, RegressorMixin, MultiOutputMixin):
    """
    A scikit-learn implementation of "BOOMER", an algorithm for learning gradient boosted multi-label classification
    rules.
    """

    def __init__(self,
                 random_state: int = 1,
                 feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value,
                 prediction_format: str = SparsePolicy.AUTO.value,
                 statistic_format: Optional[str] = None,
                 default_rule: Optional[str] = None,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 global_pruning: Optional[str] = None,
                 sequential_post_optimization: Optional[str] = None,
                 head_type: Optional[str] = None,
                 loss: Optional[str] = None,
                 binary_predictor: Optional[str] = None,
                 probability_predictor: Optional[str] = None,
                 label_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 feature_binning: Optional[str] = None,
                 label_binning: Optional[str] = None,
                 rule_pruning: Optional[str] = None,
                 shrinkage: Optional[float] = 0.3,
                 l1_regularization_weight: Optional[float] = None,
                 l2_regularization_weight: Optional[float] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param statistic_format:                The format to be used for representation of gradients and Hessians. Must
                                                be 'dense', 'sparse' or 'auto', if the most suitable format should be
                                                chosen automatically
        :param default_rule:                    Whether a default rule should be induced or not. Must be 'true', 'false'
                                                or 'auto', if it should be decided automatically whether a default rule
                                                should be induced or not
        :param rule_induction:                  The algorithm that should be used for the induction of individual rules.
                                                Must be 'top-down-greedy' or 'top-down-beam-search'. For additional
                                                options refer to the documentation
        :param max_rules:                       The maximum number of rules to be learned (including the default rule).
                                                Must be at least 1 or 0, if the number of rules should not be restricted
        :param time_limit:                      The duration in seconds after which the induction of rules should be
                                                canceled. Must be at least 1 or 0, if no time limit should be set
        :param global_pruning:                  The strategy that should be used for pruning entire rules. Must be
                                                'pre-pruning', 'post-pruning' or 'none', if no pruning should be used.
                                                For additional options refer to the documentation
        :param sequential_post_optimization:    Whether each rule in a previously learned model should be optimized by
                                                being relearned in the context of the other rules or not. Must be 'true'
                                                or 'false'. For additional options refer to the documentation
        :param head_type:                       The type of the rule heads that should be used. Must be 'single-label',
                                                'complete', 'partial-fixed', 'partial-dynamic' or 'auto', if the type of
                                                the heads should be chosen automatically. For additional options refer
                                                to the documentation
        :param loss:                            The loss function to be minimized. Must be 'squared-error-label-wise',
                                                'squared-error-example-wise', 'squared-hinge-label-wise',
                                                'squared-hinge-example-wise', 'logistic-label-wise' or
                                                'logistic-example-wise'
        :param binary_predictor:                The strategy that should be used for predicting binary labels. Must be
                                                'label-wise', 'example-wise', 'gfm' or 'auto', if the most suitable
                                                strategy should be chosen automatically, depending on the loss function
        :param probability_predictor:           The strategy that should be used for predicting probabilities. Must be
                                                'label-wise', 'marginalized' or 'auto', if the most suitable strategy
                                                should be chosen automatically, depending on the loss function
        :param label_sampling:                  The strategy that should be used to sample from the available labels
                                                whenever a new rule is learned. Must be 'without-replacement' or 'none',
                                                if no sampling should be used. For additional options refer to the
                                                documentation
        :param instance_sampling:               The strategy that should be used to sample from the available the
                                                training examples whenever a new rule is learned. Must be
                                                'with-replacement', 'without-replacement', 'stratified_label_wise',
                                                'stratified_example_wise' or 'none', if no sampling should be used. For
                                                additional options refer to the documentation
        :param feature_sampling:                The strategy that is used to sample from the available features whenever
                                                a rule is refined. Must be 'without-replacement' or 'none', if no
                                                sampling should be used. For additional options refer to the
                                                documentation
        :param holdout:                         The name of the strategy that should be used to create a holdout set.
                                                Must be 'random', 'stratified-label-wise', 'stratified-example-wise' or
                                                'none', if no holdout set should be used. If set to 'auto', the most
                                                suitable strategy is chosen automatically depending on whether a holdout
                                                set is needed and depending on the loss function. For additional options
                                                refer to the documentation
        :param feature_binning:                 The strategy that should be used to assign examples to bins based on
                                                their feature values. Must be 'auto', 'equal-width', 'equal-frequency'
                                                or 'none', if no feature binning should be used. If set to 'auto', the
                                                most suitable strategy is chosen automatically, depending on the
                                                characteristics of the feature matrix. For additional options refer to
                                                the documentation
        :param label_binning:                   The strategy that should be used to assign labels to bins. Must be
                                                'auto', 'equal-width' or 'none', if no label binning should be used. If
                                                set to 'auto', the most suitable strategy is chosen automatically,
                                                depending on the loss function and the type of rule heads. For
                                                additional options refer to the documentation
        :param rule_pruning:                    The strategy that should be used to prune individual rules. Must be
                                                'irep' or 'none', if no pruning should be used
        :param shrinkage:                       The shrinkage parameter, a.k.a. the "learning rate", that should be used
                                                to shrink the weight of individual rules. Must be in (0, 1]
        :param l1_regularization_weight:        The weight of the L1 regularization. Must be at least 0
        :param l2_regularization_weight:        The weight of the L2 regularization. Must be at least 0
        :param parallel_rule_refinement:        Whether potential refinements of rules should be searched for in
                                                parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                                strategy should be chosen automatically depending on the loss function.
                                                For additional options refer to the documentation
        :param parallel_statistic_update:       Whether the gradients and Hessians for different examples should be
                                                updated in parallel or not. Must be 'true', 'false' or 'auto', if the
                                                most suitable strategy should be chosen automatically, depending on the
                                                loss function. For additional options refer to the documentation
        :param parallel_prediction:             Whether predictions for different examples should be obtained in
                                                parallel or not. Must be 'true' or 'false'. For additional options refer
                                                to the documentation
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.statistic_format = statistic_format
        self.default_rule = default_rule
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.global_pruning = global_pruning
        self.sequential_post_optimization = sequential_post_optimization
        self.head_type = head_type
        self.loss = loss
        self.binary_predictor = binary_predictor
        self.probability_predictor = probability_predictor
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.rule_pruning = rule_pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def _create_learner(self) -> RuleLearnerWrapper:
        config = BoomerConfig()
        configure_rule_induction(config, get_string(self.rule_induction))
        configure_label_sampling(config, get_string(self.label_sampling))
        configure_instance_sampling(config, get_string(self.instance_sampling))
        configure_feature_sampling(config, get_string(self.feature_sampling))
        configure_rule_pruning(config, get_string(self.rule_pruning))
        configure_parallel_prediction(config, get_string(self.parallel_prediction))
        configure_size_stopping_criterion(config, max_rules=get_int(self.max_rules))
        configure_time_stopping_criterion(config, time_limit=get_int(self.time_limit))
        configure_global_pruning(config, get_string(self.global_pruning))
        configure_sequential_post_optimization(config, get_string(self.sequential_post_optimization))
        configure_post_processor(config, shrinkage=get_float(self.shrinkage))
        configure_l1_regularization(config, l1_regularization_weight=get_float(self.l1_regularization_weight))
        configure_l2_regularization(config, l2_regularization_weight=get_float(self.l2_regularization_weight))
        self.__configure_default_rule(config)
        self.__configure_partition_sampling(config)
        self.__configure_feature_binning(config)
        self.__configure_head_type(config)
        self.__configure_statistics(config)
        self.__configure_loss(config)
        self.__configure_label_binning(config)
        self.__configure_binary_predictor(config)
        self.__configure_probability_predictor(config)
        self.__configure_parallel_rule_refinement(config)
        self.__configure_parallel_statistic_update(config)
        return BoomerWrapper(config)

    def __configure_default_rule(self, config: BoomerConfig):
        default_rule = get_string(self.default_rule)

        if default_rule == AUTOMATIC:
            config.use_automatic_default_rule()
        else:
            configure_default_rule(config, default_rule)

    def __configure_partition_sampling(self, config: BoomerConfig):
        holdout = get_string(self.holdout)

        if holdout == AUTOMATIC:
            config.use_automatic_partition_sampling()
        else:
            configure_partition_sampling(config, holdout)

    def __configure_feature_binning(self, config: BoomerConfig):
        feature_binning = get_string(self.feature_binning)

        if feature_binning == AUTOMATIC:
            config.use_automatic_feature_binning()
        else:
            configure_feature_binning(config, feature_binning)

    def __configure_parallel_rule_refinement(self, config: BoomerConfig):
        parallel_rule_refinement = get_string(self.parallel_rule_refinement)

        if parallel_rule_refinement == AUTOMATIC:
            config.use_automatic_parallel_rule_refinement()
        else:
            configure_parallel_rule_refinement(config, parallel_rule_refinement)

    def __configure_parallel_statistic_update(self, config: BoomerConfig):
        parallel_statistic_update = get_string(self.parallel_statistic_update)

        if parallel_statistic_update == AUTOMATIC:
            config.use_automatic_parallel_statistic_update()
        else:
            configure_parallel_statistic_update(config, parallel_statistic_update)

    def __configure_head_type(self, config: BoomerConfig):
        head_type = get_string(self.head_type)

        if head_type == AUTOMATIC:
            config.use_automatic_heads()
        else:
            configure_head_type(config, head_type)

    def __configure_statistics(self, config: BoomerConfig):
        statistic_format = get_string(self.statistic_format)

        if statistic_format == AUTOMATIC:
            config.use_automatic_statistics()
        else:
            configure_statistics(config, statistic_format)

    def __configure_loss(self, config: BoomerConfig):
        loss = get_string(self.loss)

        if loss is not None:
            value = parse_param('loss', loss, LOSS_VALUES)
            configure_label_wise_squared_error_loss(config, value)
            configure_label_wise_squared_hinge_loss(config, value)
            configure_label_wise_logistic_loss(config, value)
            configure_example_wise_squared_error_loss(config, value)
            configure_example_wise_squared_hinge_loss(config, value)
            configure_example_wise_logistic_loss(config, value)

    def __configure_label_binning(self, config: BoomerConfig):
        label_binning = get_string(self.label_binning)

        if label_binning == AUTOMATIC:
            config.use_automatic_label_binning()
        else:
            configure_label_binning(config, label_binning)

    def __configure_binary_predictor(self, config: BoomerConfig):
        binary_predictor = get_string(self.binary_predictor)

        if binary_predictor == AUTOMATIC:
            config.use_automatic_binary_predictor()
        elif binary_predictor is not None:
            value, options = parse_param_and_options('binary_predictor', binary_predictor, BINARY_PREDICTOR_VALUES)
            configure_label_wise_binary_predictor(config, value, options)
            configure_example_wise_binary_predictor(config, value, options)
            configure_gfm_binary_predictor(config, value, options)

    def __configure_probability_predictor(self, config: BoomerConfig):
        probability_predictor = get_string(self.probability_predictor)

        if probability_predictor == AUTOMATIC:
            config.use_automatic_probability_predictor()
        elif probability_predictor is not None:
            value = parse_param('probability_predictor', probability_predictor, PROBABILITY_PREDICTOR_VALUES)
            configure_label_wise_probability_predictor(config, value)
            configure_marginalized_probability_predictor(config, value)
