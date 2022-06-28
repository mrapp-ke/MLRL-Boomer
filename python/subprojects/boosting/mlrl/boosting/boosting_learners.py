"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
from typing import Dict, Set, Optional

from mlrl.boosting.cython.learner import BoostingRuleLearnerConfig
from mlrl.boosting.cython.learner_boomer import Boomer as BoomerWrapper, BoomerConfig
from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, NONE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_NUM_THREADS, BINNING_EQUAL_WIDTH, BINNING_EQUAL_FREQUENCY
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import configure_rule_induction, \
    configure_feature_binning, configure_label_sampling, configure_instance_sampling, configure_feature_sampling, \
    configure_partition_sampling, configure_pruning, configure_parallel_rule_refinement, \
    configure_parallel_statistic_update, configure_parallel_prediction, configure_size_stopping_criterion, \
    configure_time_stopping_criterion, configure_early_stopping_criterion
from mlrl.common.rule_learners import parse_param, parse_param_and_options, get_string, get_int, get_float
from sklearn.base import ClassifierMixin

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

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

NON_DECOMPOSABLE_LOSSES = {LOSS_LOGISTIC_EXAMPLE_WISE}

CLASSIFICATION_PREDICTOR_LABEL_WISE = 'label-wise'

CLASSIFICATION_PREDICTOR_EXAMPLE_WISE = 'example-wise'

PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

STATISTIC_FORMAT_VALUES: Set[str] = {
    STATISTIC_FORMAT_DENSE,
    STATISTIC_FORMAT_SPARSE,
    AUTOMATIC
}

DEFAULT_RULE_VALUES: Set[str] = {
    BooleanOption.TRUE.value,
    BooleanOption.FALSE.value,
    AUTOMATIC
}

HEAD_TYPE_VALUES: Dict[str, Set[str]] = {
    HEAD_TYPE_SINGLE: {},
    HEAD_TYPE_PARTIAL_FIXED: {ARGUMENT_LABEL_RATIO, ARGUMENT_MIN_LABELS, ARGUMENT_MAX_LABELS},
    HEAD_TYPE_PARTIAL_DYNAMIC: {ARGUMENT_THRESHOLD, ARGUMENT_EXPONENT},
    HEAD_TYPE_COMPLETE: {},
    AUTOMATIC: {}
}

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_FREQUENCY: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {},
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {}
}

LOSS_VALUES: Set[str] = {
    LOSS_SQUARED_ERROR_LABEL_WISE,
    LOSS_SQUARED_HINGE_LABEL_WISE,
    LOSS_LOGISTIC_LABEL_WISE,
    LOSS_LOGISTIC_EXAMPLE_WISE
}

CLASSIFICATION_PREDICTOR_VALUES: Set[str] = {
    CLASSIFICATION_PREDICTOR_LABEL_WISE,
    CLASSIFICATION_PREDICTOR_EXAMPLE_WISE,
    AUTOMATIC
}

PROBABILITY_PREDICTOR_VALUES: Set[str] = {
    PROBABILITY_PREDICTOR_LABEL_WISE,
    PROBABILITY_PREDICTOR_MARGINALIZED,
    AUTOMATIC
}

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {},
    AUTOMATIC: {}
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


class Boomer(MLRuleLearner, ClassifierMixin):
    """
    A scikit-learn implementation of "BOOMER", an algorithm for learning gradient boosted multi-label classification
    rules.
    """

    def __init__(self,
                 random_state: int = 1,
                 feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value,
                 predicted_label_format: str = SparsePolicy.AUTO.value,
                 statistic_format: Optional[str] = None,
                 default_rule: Optional[str] = None,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 early_stopping: Optional[str] = None,
                 head_type: Optional[str] = None,
                 loss: Optional[str] = None,
                 classification_predictor: Optional[str] = None,
                 probability_predictor: Optional[str] = None,
                 label_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 feature_binning: Optional[str] = None,
                 label_binning: Optional[str] = None,
                 pruning: Optional[str] = None,
                 shrinkage: Optional[float] = 0.3,
                 l1_regularization_weight: Optional[float] = None,
                 l2_regularization_weight: Optional[float] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param statistic_format:            The format to be used for representation of gradients and Hessians. Must be
                                            'dense', 'sparse' or 'auto', if the most suitable format should be chosen
                                            automatically
        :param default_rule:                Whether a default rule should be induced or not. Must be 'true', 'false' or
                                            'auto', if it should be decided automatically whether a default rule should
                                            be induced or not
        :param rule_induction:              The algorithm that should be used for the induction of individual rules.
                                            Must be 'top-down-greedy' or 'top-down-beam-search'. For additional options
                                            refer to the documentation
        :param max_rules:                   The maximum number of rules to be learned (including the default rule). Must
                                            be at least 1 or 0, if the number of rules should not be restricted.
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled. Must be at least 1 or 0, if no time limit should be set
        :param early_stopping:              The strategy that should be used for early stopping. Must be 'loss', if the
                                            induction of new rules should be stopped as soon as the performance of the
                                            model does not improve on a holdout set according to the loss function or
                                            'none', if no early stopping should be used. For additional options refer to
                                            the documentation
        :param head_type:                   The type of the rule heads that should be used. Must be 'single-label',
                                            'complete', 'partial-fixed', 'partial-dynamic' or 'auto', if the type of the
                                            heads should be chosen automatically. For additional options refer to the
                                            documentation
        :param loss:                        The loss function to be minimized. Must be 'squared-error-label-wise',
                                            'squared-hinge-label-wise', 'logistic-label-wise' or 'logistic-example-wise'
        :param classification_predictor:    The strategy that should be used for predicting binary labels. Must be
                                            'label-wise', 'example-wise' or 'auto', if the most suitable strategy should
                                            be chosen automatically, depending on the loss function
        :param probability_predictor:       The strategy that should be used for predicting probabilities. Must be
                                            'label-wise', 'marginalized' or 'auto', if the most suitable strategy should
                                            be chosen automatically, depending on the loss function
        :param label_sampling:              The strategy that should be used to sample from the available labels
                                            whenever a new rule is learned. Must be 'without-replacement' or 'none', if
                                            no sampling should be used. For additional options refer to the
                                            documentation
        :param instance_sampling:           The strategy that should be used to sample from the available the training
                                            examples whenever a new rule is learned. Must be 'with-replacement',
                                            'without-replacement', 'stratified_label_wise', 'stratified_example_wise' or
                                            'none', if no sampling should be used. For additional options refer to the
                                            documentation
        :param feature_sampling:            The strategy that is used to sample from the available features whenever a
                                            rule is refined. Must be 'without-replacement' or 'none', if no sampling
                                            should be used. For additional options refer to the documentation
        :param holdout:                     The name of the strategy that should be used to creating a holdout set. Must
                                            be 'random', 'stratified-label-wise', 'stratified-example-wise' or 'none',
                                            if no holdout set should be used. For additional options refer to the
                                            documentation
        :param feature_binning:             The strategy that should be used to assign examples to bins based on their
                                            feature values. Must be 'auto', 'equal-width', 'equal-frequency' or 'none',
                                            if no feature binning should be used. If set to 'auto', the most suitable
                                            strategy is chosen automatically, depending on the characteristics of the
                                            feature matrix. For additional options refer to the documentation
        :param label_binning:               The strategy that should be used to assign labels to bins. Must be 'auto',
                                            'equal-width' or 'none', if no label binning should be used. If set to
                                            'auto', the most suitable strategy is chosen automatically, depending on the
                                            loss function and the type of rule heads. For additional options refer to
                                            the documentation
        :param pruning:                     The strategy that should be used to prune individual rules. Must be 'irep'
                                            or 'none', if no pruning should be used
        :param shrinkage:                   The shrinkage parameter, a.k.a. the "learning rate", that should be used to
                                            shrink the weight of individual rules. Must be in (0, 1]
        :param l1_regularization_weight:    The weight of the L1 regularization. Must be at least 0
        :param l2_regularization_weight:    The weight of the L2 regularization. Must be at least 0
        :param parallel_rule_refinement:    Whether potential refinements of rules should be searched for in parallel or
                                            not. Must be 'true', 'false' or 'auto', if the most suitable strategy should
                                            be chosen automatically depending on the loss function. For additional
                                            options refer to the documentation
        :param parallel_statistic_update:   Whether the gradients and Hessians for different examples should be updated
                                            in parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                            strategy should be chosen automatically, depending on the loss function. For
                                            additional options refer to the documentation
        :param parallel_prediction:         Whether predictions for different examples should be obtained in parallel or
                                            not. Must be 'true' or 'false'. For additional options refer to the
                                            documentation
        """
        super().__init__(random_state, feature_format, label_format, predicted_label_format)
        self.statistic_format = statistic_format
        self.default_rule = default_rule
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.early_stopping = early_stopping
        self.head_type = head_type
        self.loss = loss
        self.classification_predictor = classification_predictor
        self.probability_predictor = probability_predictor
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def _create_learner(self) -> RuleLearnerWrapper:
        config = BoomerConfig()
        self.__configure_default_rule(config)
        configure_rule_induction(config, get_string(self.rule_induction))
        self.__configure_feature_binning(config)
        configure_label_sampling(config, get_string(self.label_sampling))
        configure_instance_sampling(config, get_string(self.instance_sampling))
        configure_feature_sampling(config, get_string(self.feature_sampling))
        configure_partition_sampling(config, get_string(self.holdout))
        configure_pruning(config, get_string(self.pruning))
        self.__configure_parallel_rule_refinement(config)
        self.__configure_parallel_statistic_update(config)
        configure_parallel_prediction(config, get_string(self.parallel_prediction))
        configure_size_stopping_criterion(config, max_rules=get_int(self.max_rules))
        configure_time_stopping_criterion(config, time_limit=get_int(self.time_limit))
        configure_early_stopping_criterion(config, get_string(self.early_stopping))
        configure_post_processor(config, shrinkage=get_float(self.shrinkage))
        self.__configure_head_type(config)
        self.__configure_statistics(config)
        configure_l1_regularization(config, l1_regularization_weight=get_float(self.l1_regularization_weight))
        configure_l2_regularization(config, l2_regularization_weight=get_float(self.l2_regularization_weight))
        self.__configure_loss(config)
        self.__configure_label_binning(config)
        self.__configure_classification_predictor(config)
        self.__configure_probability_predictor(config)
        return BoomerWrapper(config)

    def __configure_default_rule(self, config: BoomerConfig):
        default_rule = get_string(self.default_rule)

        if default_rule is not None:
            value = parse_param('default_rule', default_rule, DEFAULT_RULE_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_default_rule()
            elif value == BooleanOption.TRUE.value:
                config.use_default_rule()
            else:
                config.use_no_default_rule()

    def __configure_feature_binning(self, config: BoomerConfig):
        feature_binning = get_string(self.feature_binning)

        if feature_binning is not None:
            if feature_binning == AUTOMATIC:
                config.use_automatic_feature_binning()
            else:
                configure_feature_binning(config, feature_binning)

    def __configure_parallel_rule_refinement(self, config: BoomerConfig):
        parallel_rule_refinement = get_string(self.parallel_rule_refinement)

        if parallel_rule_refinement is not None:
            if parallel_rule_refinement == AUTOMATIC:
                config.use_automatic_parallel_rule_refinement()
            else:
                configure_parallel_rule_refinement(config, parallel_rule_refinement)

    def __configure_parallel_statistic_update(self, config: BoomerConfig):
        parallel_statistic_update = get_string(self.parallel_statistic_update)

        if parallel_statistic_update is not None:
            if parallel_statistic_update == AUTOMATIC:
                config.use_automatic_parallel_statistic_update()
            else:
                configure_parallel_statistic_update(config, parallel_statistic_update)

    def __configure_head_type(self, config: BoomerConfig):
        head_type = get_string(self.head_type)

        if head_type is not None:
            value, options = parse_param_and_options("head_type", head_type, HEAD_TYPE_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_heads()
            elif value == HEAD_TYPE_SINGLE:
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

    def __configure_statistics(self, config: BoomerConfig):
        statistic_format = get_string(self.statistic_format)

        if statistic_format is not None:
            value = parse_param("statistic_format", statistic_format, STATISTIC_FORMAT_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_statistics()
            elif value == STATISTIC_FORMAT_DENSE:
                config.use_dense_statistics()
            elif value == STATISTIC_FORMAT_SPARSE:
                config.use_sparse_statistics()

    def __configure_loss(self, config: BoomerConfig):
        loss = get_string(self.loss)

        if loss is not None:
            value = parse_param("loss", loss, LOSS_VALUES)

            if value == LOSS_SQUARED_ERROR_LABEL_WISE:
                config.use_label_wise_squared_error_loss()
            elif value == LOSS_SQUARED_HINGE_LABEL_WISE:
                config.use_label_wise_squared_hinge_loss()
            elif value == LOSS_LOGISTIC_LABEL_WISE:
                config.use_label_wise_logistic_loss()
            elif value == LOSS_LOGISTIC_EXAMPLE_WISE:
                config.use_example_wise_logistic_loss()

    def __configure_label_binning(self, config: BoomerConfig):
        label_binning = get_string(self.label_binning)

        if label_binning is not None:
            value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

            if value == NONE:
                config.use_no_label_binning()
            elif value == AUTOMATIC:
                config.use_automatic_label_binning()
            if value == BINNING_EQUAL_WIDTH:
                c = config.use_equal_width_label_binning()
                c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
                c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
                c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))

    def __configure_classification_predictor(self, config: BoomerConfig):
        classification_predictor = get_string(self.classification_predictor)

        if classification_predictor is not None:
            value = parse_param('classification_predictor', classification_predictor, CLASSIFICATION_PREDICTOR_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_label_binning()
            elif value == CLASSIFICATION_PREDICTOR_LABEL_WISE:
                config.use_label_wise_classification_predictor()
            elif value == CLASSIFICATION_PREDICTOR_EXAMPLE_WISE:
                config.use_example_wise_classification_predictor()

    def __configure_probability_predictor(self, config: BoomerConfig):
        probability_predictor = get_string(self.probability_predictor)

        if probability_predictor is not None:
            value = parse_param('probability_predictor', probability_predictor, PROBABILITY_PREDICTOR_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_probability_predictor()
            elif value == PROBABILITY_PREDICTOR_LABEL_WISE:
                config.use_label_wise_probability_predictor()
            elif value == PROBABILITY_PREDICTOR_MARGINALIZED:
                config.use_marginalized_probability_predictor()
