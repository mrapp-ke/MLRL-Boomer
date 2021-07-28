#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides scikit-learn implementations of boosting algorithms.
"""
import logging as log
from typing import Optional, Dict, Set

from mlrl.boosting.cython.binning import LabelBinningFactory, EqualWidthLabelBinningFactory
from mlrl.boosting.cython.losses_example_wise import ExampleWiseLogisticLoss
from mlrl.boosting.cython.losses_label_wise import LabelWiseLoss, LabelWiseLogisticLoss, LabelWiseSquaredErrorLoss, \
    LabelWiseSquaredHingeLoss
from mlrl.boosting.cython.model import RuleListBuilder
from mlrl.boosting.cython.output import LabelWiseClassificationPredictor, ExampleWiseClassificationPredictor, \
    LabelWiseProbabilityPredictor, LabelWiseTransformationFunction, LogisticFunction
from mlrl.boosting.cython.post_processing import ConstantShrinkage
from mlrl.boosting.cython.rule_evaluation_example_wise import RegularizedExampleWiseRuleEvaluationFactory, \
    BinnedExampleWiseRuleEvaluationFactory
from mlrl.boosting.cython.rule_evaluation_label_wise import RegularizedLabelWiseRuleEvaluationFactory, \
    BinnedLabelWiseRuleEvaluationFactory
from mlrl.boosting.cython.statistics_example_wise import DenseExampleWiseStatisticsProviderFactory
from mlrl.boosting.cython.statistics_label_wise import DenseLabelWiseStatisticsProviderFactory
from mlrl.common.cython.head_refinement import HeadRefinementFactory, NoHeadRefinementFactory, \
    SingleLabelHeadRefinementFactory, CompleteHeadRefinementFactory
from mlrl.common.cython.input import LabelMatrix, LabelVectorSet
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.post_processing import PostProcessor, NoPostProcessor
from mlrl.common.cython.rule_induction import TopDownRuleInduction
from mlrl.common.cython.rule_model_assemblage import SequentialRuleModelAssemblage
from mlrl.common.cython.sampling import InstanceSamplingFactory, NoInstanceSamplingFactory, \
    InstanceSamplingWithReplacementFactory, InstanceSamplingWithoutReplacementFactory, \
    LabelWiseStratifiedSamplingFactory, ExampleWiseStratifiedSamplingFactory
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.common.cython.stopping import MeasureStoppingCriterion, AggregationFunction, MinFunction, MaxFunction, \
    ArithmeticMeanFunction
from sklearn.base import ClassifierMixin

from mlrl.common.rule_learners import AUTOMATIC, SAMPLING_WITH_REPLACEMENT, SAMPLING_WITHOUT_REPLACEMENT, \
    SAMPLING_STRATIFIED_LABEL_WISE, SAMPLING_STRATIFIED_EXAMPLE_WISE, HEAD_TYPE_SINGLE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_SAMPLE_SIZE
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import create_pruning, create_feature_sampling_factory, create_label_sampling_factory, \
    create_partition_sampling_factory, create_stopping_criteria, get_preferred_num_threads, create_thresholds_factory, \
    parse_param, parse_param_and_options

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

LOSS_SQUARED_HINGE_LABEL_WISE = 'hinge-label-wise'

NON_DECOMPOSABLE_LOSSES = {LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_LABEL_WISE = 'label-wise'

PREDICTOR_EXAMPLE_WISE = 'example-wise'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

INSTANCE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITH_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_SAMPLE_SIZE}
}

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


class Boomer(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of "BOOMER", an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, max_rules: int = 1000, default_rule: bool = True,
                 time_limit: int = -1, early_stopping: str = None, head_type: str = AUTOMATIC,
                 loss: str = LOSS_LOGISTIC_LABEL_WISE, predictor: str = AUTOMATIC, label_sampling: str = None,
                 instance_sampling: str = None, recalculate_predictions: bool = True,
                 feature_sampling: str = SAMPLING_WITHOUT_REPLACEMENT, holdout: str = None, feature_binning: str = None,
                 label_binning: str = AUTOMATIC, pruning: str = None, shrinkage: float = 0.3,
                 l2_regularization_weight: float = 1.0, min_coverage: int = 1, max_conditions: int = -1,
                 max_head_refinements: int = 1, num_threads_rule_refinement: int = 1,
                 num_threads_statistic_update: int = 1, num_threads_prediction: int = 1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param default_rule:                        True, if the first rule should be a default rule, False otherwise
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
                                                    `with-replacement`, `without-replacement` or None, if no sampling
                                                    should be used. Additional options may be provided using the bracket
                                                    notation `with-replacement{sample_size=0.5}`
        :param recalculate_predictions:             True, if the predictions of rules should be recalculated on the
                                                    entire training data, if instance sampling is used, False otherwise
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
        :param l2_regularization_weight:            The weight of the L2 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    -1, if the number of refinements should not be restricted
        :param num_threads_rule_refinement:         The number of threads to be used to search for potential refinements
                                                    of rules or -1, if the number of cores that are available on the
                                                    machine should be used
        :param num_threads_statistic_update:        The number of threads to be used to update statistics or -1, if the
                                                    number of cores that are available on the machine should be used
        :param num_threads_prediction:              The number of threads to be used to make predictions or -1, if the
                                                    number of cores that are available on the machine should be used
        """
        super().__init__(random_state, feature_format, label_format)
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
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.num_threads_rule_refinement = num_threads_rule_refinement
        self.num_threads_statistic_update = num_threads_statistic_update
        self.num_threads_prediction = num_threads_prediction

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

    def _create_predictor(self, num_labels: int) -> Predictor:
        predictor = self.__get_preferred_predictor()
        value = parse_param('predictor', predictor, PREDICTOR_VALUES)

        if value == PREDICTOR_LABEL_WISE:
            return self.__create_label_wise_predictor(num_labels)
        elif value == PREDICTOR_EXAMPLE_WISE:
            return self.__create_example_wise_predictor(num_labels)

    def _create_probability_predictor(self, num_labels: int) -> Predictor:
        predictor = self.__get_preferred_predictor()

        if self.loss == LOSS_LOGISTIC_LABEL_WISE or self.loss == LOSS_LOGISTIC_EXAMPLE_WISE:
            if predictor == PREDICTOR_LABEL_WISE:
                transformation_function = LogisticFunction()
                return self.__create_label_wise_probability_predictor(num_labels, transformation_function)
        return None

    def _create_label_vector_set(self, label_matrix: LabelMatrix) -> LabelVectorSet:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_EXAMPLE_WISE:
            return LabelVectorSet.create(label_matrix)
        return None

    def __get_preferred_predictor(self) -> str:
        predictor = self.predictor

        if predictor == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return PREDICTOR_EXAMPLE_WISE
            else:
                return PREDICTOR_LABEL_WISE
        return predictor

    def __create_label_wise_predictor(self, num_labels: int) -> LabelWiseClassificationPredictor:
        num_threads = get_preferred_num_threads(int(self.num_threads_prediction))
        threshold = 0.5 if self.loss == LOSS_SQUARED_HINGE_LABEL_WISE else 0.0
        return LabelWiseClassificationPredictor(num_labels=num_labels, threshold=threshold, num_threads=num_threads)

    def __create_example_wise_predictor(self, num_labels: int) -> ExampleWiseClassificationPredictor:
        loss = self.__create_loss_function()
        num_threads = get_preferred_num_threads(int(self.num_threads_prediction))
        return ExampleWiseClassificationPredictor(num_labels=num_labels, measure=loss, num_threads=num_threads)

    def __create_label_wise_probability_predictor(
            self, num_labels: int,
            transformation_function: LabelWiseTransformationFunction) -> LabelWiseProbabilityPredictor:
        num_threads = get_preferred_num_threads(int(self.num_threads_prediction))
        return LabelWiseProbabilityPredictor(num_labels=num_labels, transformation_function=transformation_function,
                                             num_threads=num_threads)

    def _create_model_builder(self) -> ModelBuilder:
        return RuleListBuilder()

    def _create_rule_model_assemblage(self, num_labels: int) -> SequentialRuleModelAssemblage:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        early_stopping_criterion = self.__create_early_stopping()
        if early_stopping_criterion is not None:
            stopping_criteria.append(early_stopping_criterion)
        label_sampling_factory = create_label_sampling_factory(self.label_sampling)
        instance_sampling_factory = self.__create_instance_sampling_factory()
        feature_sampling_factory = create_feature_sampling_factory(self.feature_sampling)
        partition_sampling_factory = create_partition_sampling_factory(self.holdout)
        pruning = create_pruning(self.pruning, self.instance_sampling)
        shrinkage = self.__create_post_processor()
        loss_function = self.__create_loss_function()
        default_rule_head_refinement_factory = CompleteHeadRefinementFactory() if self.default_rule \
            else NoHeadRefinementFactory()
        head_refinement_factory = self.__create_head_refinement_factory()
        l2_regularization_weight = float(self.l2_regularization_weight)
        rule_evaluation_factory = self.__create_rule_evaluation_factory(loss_function, l2_regularization_weight)
        num_threads_statistic_update = get_preferred_num_threads(int(self.num_threads_statistic_update))
        statistics_provider_factory = self.__create_statistics_provider_factory(loss_function, rule_evaluation_factory,
                                                                                num_threads_statistic_update)
        thresholds_factory = create_thresholds_factory(self.feature_binning, num_threads_statistic_update)
        num_threads_rule_refinement = get_preferred_num_threads(int(self.num_threads_rule_refinement))
        rule_induction = TopDownRuleInduction(int(self.min_coverage), int(self.max_conditions),
                                              int(self.max_head_refinements), self.recalculate_predictions,
                                              num_threads_rule_refinement)
        return SequentialRuleModelAssemblage(statistics_provider_factory, thresholds_factory, rule_induction,
                                             default_rule_head_refinement_factory, head_refinement_factory,
                                             label_sampling_factory, instance_sampling_factory,
                                             feature_sampling_factory, partition_sampling_factory, pruning, shrinkage,
                                             stopping_criteria)

    def __create_early_stopping(self) -> Optional[MeasureStoppingCriterion]:
        early_stopping = self.early_stopping

        if early_stopping is None:
            return None
        else:
            value, options = parse_param_and_options('early_stopping', early_stopping, EARLY_STOPPING_VALUES)

            if value == EARLY_STOPPING_LOSS:
                if self.holdout is None:
                    log.warning('Parameter "early_stopping" does not have any effect, because parameter "holdout" is '
                                + 'set to "None"!')
                    return None
                else:
                    loss = self.__create_loss_function()
                    aggregation_function = self.__create_aggregation_function(
                        options.get_string(ARGUMENT_AGGREGATION_FUNCTION, 'avg'))
                    min_rules = options.get_int(ARGUMENT_MIN_RULES, 100)
                    update_interval = options.get_int(ARGUMENT_UPDATE_INTERVAL, 1)
                    stop_interval = options.get_int(ARGUMENT_STOP_INTERVAL, 1)
                    num_past = options.get_int(ARGUMENT_NUM_PAST, 50)
                    num_recent = options.get_int(ARGUMENT_NUM_RECENT, 50)
                    min_improvement = options.get_float(ARGUMENT_MIN_IMPROVEMENT, 0.005)
                    force_stop = options.get_bool(ARGUMENT_FORCE_STOP, True)
                    return MeasureStoppingCriterion(loss, aggregation_function, min_rules=min_rules,
                                                    update_interval=update_interval, stop_interval=stop_interval,
                                                    num_past=num_past, num_recent=num_recent,
                                                    min_improvement=min_improvement, force_stop=force_stop)

    @staticmethod
    def __create_aggregation_function(aggregation_function: str) -> AggregationFunction:
        value = parse_param(ARGUMENT_AGGREGATION_FUNCTION, aggregation_function, {AGGREGATION_FUNCTION_MIN,
                                                                                  AGGREGATION_FUNCTION_MAX,
                                                                                  AGGREGATION_FUNCTION_ARITHMETIC_MEAN})
        if value == AGGREGATION_FUNCTION_MIN:
            return MinFunction()
        elif value == AGGREGATION_FUNCTION_MAX:
            return MaxFunction()
        elif value == AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
            return ArithmeticMeanFunction()

    def __create_loss_function(self):
        value = parse_param("loss", self.loss, LOSS_VALUES)

        if value == LOSS_SQUARED_ERROR_LABEL_WISE:
            return LabelWiseSquaredErrorLoss()
        elif value == LOSS_SQUARED_HINGE_LABEL_WISE:
            return LabelWiseSquaredHingeLoss()
        elif value == LOSS_LOGISTIC_LABEL_WISE:
            return LabelWiseLogisticLoss()
        elif value == LOSS_LOGISTIC_EXAMPLE_WISE:
            return ExampleWiseLogisticLoss()

    def __create_rule_evaluation_factory(self, loss_function, l2_regularization_weight: float):
        label_binning_factory = self.__create_label_binning_factory()

        if isinstance(loss_function, LabelWiseLoss):
            if label_binning_factory is None:
                return RegularizedLabelWiseRuleEvaluationFactory(l2_regularization_weight)
            else:
                return BinnedLabelWiseRuleEvaluationFactory(l2_regularization_weight, label_binning_factory)
        else:
            if label_binning_factory is None:
                return RegularizedExampleWiseRuleEvaluationFactory(l2_regularization_weight)
            else:
                return BinnedExampleWiseRuleEvaluationFactory(l2_regularization_weight, label_binning_factory)

    def __create_label_binning_factory(self) -> LabelBinningFactory:
        label_binning = self.__get_preferred_label_binning()

        if label_binning is None:
            return None
        else:
            value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

            if value == LABEL_BINNING_EQUAL_WIDTH:
                bin_ratio = options.get_float(ARGUMENT_BIN_RATIO, 0.04)
                min_bins = options.get_int(ARGUMENT_MIN_BINS, 1)
                max_bins = options.get_int(ARGUMENT_MAX_BINS, 0)
                return EqualWidthLabelBinningFactory(bin_ratio, min_bins, max_bins)

    def __get_preferred_label_binning(self):
        label_binning = self.label_binning

        if label_binning == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES and self.__get_preferred_head_type() == HEAD_TYPE_COMPLETE:
                return LABEL_BINNING_EQUAL_WIDTH
            else:
                return None
        return label_binning

    @staticmethod
    def __create_statistics_provider_factory(loss_function, rule_evaluation_factory,
                                             num_threads: int) -> StatisticsProviderFactory:
        if isinstance(loss_function, LabelWiseLoss):
            return DenseLabelWiseStatisticsProviderFactory(loss_function, rule_evaluation_factory,
                                                           rule_evaluation_factory, rule_evaluation_factory,
                                                           num_threads)
        else:
            return DenseExampleWiseStatisticsProviderFactory(loss_function, rule_evaluation_factory,
                                                             rule_evaluation_factory, rule_evaluation_factory,
                                                             num_threads)

    def __create_head_refinement_factory(self) -> HeadRefinementFactory:
        head_type = self.__get_preferred_head_type()
        value = parse_param("head_type", head_type, HEAD_TYPE_VALUES)

        if value == HEAD_TYPE_SINGLE:
            return SingleLabelHeadRefinementFactory()
        elif value == HEAD_TYPE_COMPLETE:
            return CompleteHeadRefinementFactory()

    def __get_preferred_head_type(self) -> str:
        head_type = self.head_type

        if head_type == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return HEAD_TYPE_COMPLETE
            else:
                return HEAD_TYPE_SINGLE
        return head_type

    def __create_post_processor(self) -> PostProcessor:
        shrinkage = float(self.shrinkage)

        if shrinkage == 1.0:
            return NoPostProcessor()
        else:
            return ConstantShrinkage(shrinkage)

    def __create_instance_sampling_factory(self) -> InstanceSamplingFactory:
        instance_sampling = self.instance_sampling

        if instance_sampling is None:
            return NoInstanceSamplingFactory()
        else:
            value, options = parse_param_and_options('instance_sampling', instance_sampling, INSTANCE_SAMPLING_VALUES)

            if value == SAMPLING_WITH_REPLACEMENT:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 1.0)
                return InstanceSamplingWithReplacementFactory(sample_size)
            elif value == SAMPLING_WITHOUT_REPLACEMENT:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 0.66)
                return InstanceSamplingWithoutReplacementFactory(sample_size)
            elif value == SAMPLING_STRATIFIED_LABEL_WISE:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 0.66)
                return LabelWiseStratifiedSamplingFactory(sample_size)
            elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 0.66)
                return ExampleWiseStratifiedSamplingFactory(sample_size)
