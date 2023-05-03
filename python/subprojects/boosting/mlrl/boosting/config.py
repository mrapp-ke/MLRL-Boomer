"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility function for configuring boosting algorithms.
"""
from mlrl.common.cython.learner import NoPostProcessorMixin, DefaultRuleMixin
from mlrl.boosting.cython.learner import AutomaticPartitionSamplingMixin, AutomaticFeatureBinningMixin, \
    AutomaticParallelRuleRefinementMixin, AutomaticParallelStatisticUpdateMixin, ConstantShrinkageMixin, \
    NoL1RegularizationMixin, L1RegularizationMixin, NoL2RegularizationMixin, L2RegularizationMixin, \
    DenseStatisticsMixin, SparseStatisticsMixin, AutomaticStatisticsMixin, NoDefaultRuleMixin, \
    AutomaticDefaultRuleMixin, NoLabelBinningMixin, EqualWidthLabelBinningMixin, AutomaticLabelBinningMixin, \
    LabelWiseLogisticLossMixin, ExampleWiseLogisticLossMixin, LabelWiseSquaredErrorLossMixin, \
    ExampleWiseSquaredErrorLossMixin, LabelWiseSquaredHingeLossMixin, ExampleWiseSquaredHingeLossMixin, \
    SingleLabelHeadMixin, FixedPartialHeadMixin, DynamicPartialHeadMixin, CompleteHeadMixin, \
    LabelWiseBinaryPredictorMixin, ExampleWiseBinaryPredictorMixin, GfmBinaryPredictorMixin, \
    AutomaticBinaryPredictorMixin, LabelWiseProbabilityPredictorMixin, MarginalizedProbabilityPredictorMixin, \
    AutomaticProbabilityPredictorMixin
from mlrl.common.config import FloatParameter, NominalParameter, PartitionSamplingParameter, FeatureBinningParameter, \
    ParallelRuleRefinementParameter, ParallelStatisticUpdateParameter, RULE_LEARNER_PARAMETERS, NONE, AUTOMATIC, \
    OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS, BINNING_EQUAL_WIDTH
from mlrl.common.options import Options, BooleanOption, parse_param, parse_param_and_options
from typing import Dict, Set, Optional

PARAM_DEFAULT_RULE = 'default_rule'

PARAM_LOSS = 'loss'

PARAM_HEAD_TYPE = 'head_type'

STATISTIC_FORMAT_DENSE = 'dense'

STATISTIC_FORMAT_SPARSE = 'sparse'

HEAD_TYPE_SINGLE = 'single-label'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

OPTION_LABEL_RATIO = 'label_ratio'

OPTION_MIN_LABELS = 'min_labels'

OPTION_MAX_LABELS = 'max_labels'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

OPTION_THRESHOLD = 'threshold'

OPTION_EXPONENT = 'exponent'

HEAD_TYPE_COMPLETE = 'complete'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

LOSS_SQUARED_HINGE_EXAMPLE_WISE = 'squared-hinge-example-wise'

BINARY_PREDICTOR_LABEL_WISE = 'label-wise'

BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

BINARY_PREDICTOR_GFM = 'gfm'

PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

STATISTIC_FORMAT_VALUES: Set[str] = {STATISTIC_FORMAT_DENSE, STATISTIC_FORMAT_SPARSE}

DEFAULT_RULE_VALUES: Set[str] = {BooleanOption.TRUE.value, BooleanOption.FALSE.value}

HEAD_TYPE_VALUES: Dict[str, Set[str]] = {
    HEAD_TYPE_SINGLE: {},
    HEAD_TYPE_PARTIAL_FIXED: {OPTION_LABEL_RATIO, OPTION_MIN_LABELS, OPTION_MAX_LABELS},
    HEAD_TYPE_PARTIAL_DYNAMIC: {OPTION_THRESHOLD, OPTION_EXPONENT},
    HEAD_TYPE_COMPLETE: {}
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_WIDTH: {OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS}
}


def configure_post_processor(config, shrinkage: Optional[float]):
    if shrinkage is not None:
        if shrinkage == 1:
            config.use_no_post_processor()
        else:
            config.use_constant_shrinkage_post_processor().set_shrinkage(shrinkage)


def configure_l1_regularization(config, l1_regularization_weight: Optional[float]):
    if l1_regularization_weight is not None:
        if l1_regularization_weight == 0:
            config.use_no_l1_regularization()
        else:
            config.use_l1_regularization().set_regularization_weight(l1_regularization_weight)


def configure_l2_regularization(config, l2_regularization_weight: Optional[float]):
    if l2_regularization_weight is not None:
        if l2_regularization_weight == 0:
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(l2_regularization_weight)


def configure_default_rule(config, default_rule: Optional[str]):
    if default_rule is not None:
        value = parse_param('default_rule', default_rule, DEFAULT_RULE_VALUES)

        if value == BooleanOption.TRUE.value:
            config.use_default_rule()
        else:
            config.use_no_default_rule()


def configure_head_type(config, head_type: Optional[str]):
    if head_type is not None:
        value, options = parse_param_and_options("head_type", head_type, HEAD_TYPE_VALUES)

        if value == HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == HEAD_TYPE_PARTIAL_FIXED:
            c = config.use_fixed_partial_heads()
            c.set_label_ratio(options.get_float(OPTION_LABEL_RATIO, c.get_label_ratio()))
            c.set_min_labels(options.get_int(OPTION_MIN_LABELS, c.get_min_labels()))
            c.set_max_labels(options.get_int(OPTION_MAX_LABELS, c.get_max_labels()))
        elif value == HEAD_TYPE_PARTIAL_DYNAMIC:
            c = config.use_dynamic_partial_heads()
            c.set_threshold(options.get_float(OPTION_THRESHOLD, c.get_threshold()))
            c.set_exponent(options.get_float(OPTION_EXPONENT, c.get_exponent()))
        elif value == HEAD_TYPE_COMPLETE:
            config.use_complete_heads()


def configure_statistics(config, statistic_format: Optional[str]):
    if statistic_format is not None:
        value = parse_param("statistic_format", statistic_format, STATISTIC_FORMAT_VALUES)

        if value == STATISTIC_FORMAT_DENSE:
            config.use_dense_statistics()
        elif value == STATISTIC_FORMAT_SPARSE:
            config.use_sparse_statistics()


def configure_label_binning(config, label_binning: Optional[str]):
    if label_binning is not None:
        value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

        if value == NONE:
            config.use_no_label_binning()
        if value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_label_binning()
            c.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(OPTION_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(OPTION_MAX_BINS, c.get_max_bins()))


def configure_label_wise_squared_error_loss(config, value: str):
    if value == LOSS_SQUARED_ERROR_LABEL_WISE:
        config.use_label_wise_squared_error_loss()


def configure_label_wise_squared_hinge_loss(config, value: str):
    if value == LOSS_SQUARED_HINGE_LABEL_WISE:
        config.use_label_wise_squared_hinge_loss()


def configure_label_wise_logistic_loss(config, value: str):
    if value == LOSS_LOGISTIC_LABEL_WISE:
        config.use_label_wise_logistic_loss()


def configure_example_wise_logistic_loss(config, value: str):
    if value == LOSS_LOGISTIC_EXAMPLE_WISE:
        config.use_example_wise_logistic_loss()


def configure_example_wise_squared_error_loss(config, value: str):
    if value == LOSS_SQUARED_ERROR_EXAMPLE_WISE:
        config.use_example_wise_squared_error_loss()


def configure_example_wise_squared_hinge_loss(config, value: str):
    if value == LOSS_SQUARED_HINGE_EXAMPLE_WISE:
        config.use_example_wise_squared_hinge_loss()


def configure_label_wise_binary_predictor(config, value: str):
    if value == BINARY_PREDICTOR_LABEL_WISE:
        config.use_label_wise_binary_predictor()


def configure_example_wise_binary_predictor(config, value: str):
    if value == BINARY_PREDICTOR_EXAMPLE_WISE:
        config.use_example_wise_binary_predictor()


def configure_gfm_binary_predictor(config, value: str):
    if value == BINARY_PREDICTOR_GFM:
        config.use_gfm_binary_predictor()


def configure_label_wise_probability_predictor(config, value: str):
    if value == PROBABILITY_PREDICTOR_LABEL_WISE:
        config.use_label_wise_probability_predictor()


def configure_marginalized_probability_predictor(config, value: str):
    if value == PROBABILITY_PREDICTOR_MARGINALIZED:
        config.use_marginalized_probability_predictor()


class ExtendedPartitionSamplingParameter(PartitionSamplingParameter):
    """
    Extends the `PartitionSamplingParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(
            name=AUTOMATIC,
            mixin=AutomaticPartitionSamplingMixin,
            description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically depending '
            + 'on whether a holdout set is needed and depending on the loss function')

    def _configure(config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_partition_sampling()
        else:
            super()._configure(config, value, options)


class ExtendedFeatureBinningParameter(FeatureBinningParameter):
    """
    Extends the `FeatureBinningParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(
            name=AUTOMATIC,
            mixin=AutomaticFeatureBinningMixin,
            description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically based on '
            + 'the characteristics of the feature matrix')

    def _configure(config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_feature_binning()
        else:
            super()._configure(config, value, options)


class ExtendedParallelRuleRefinementParameter(ParallelRuleRefinementParameter):
    """
    Extends the `ParallelRuleRefinementParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(
            name=AUTOMATIC,
            mixin=AutomaticParallelRuleRefinementMixin,
            description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically based on '
            + 'the parameter ' + PARAM_LOSS)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_parallel_rule_refinement()
        else:
            super()._configure(config, value, options)


class ExtendedParallelStatisticUpdateParameter(ParallelStatisticUpdateParameter):
    """
    Extends the `ParallelStatisticUpdateParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(
            name=AUTOMATIC,
            mixin=AutomaticParallelStatisticUpdateMixin,
            description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically based on '
            + 'the parameter ' + PARAM_LOSS)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_parallel_statistic_update()
        else:
            super()._configure(config, value, options)


class ShrinkageParameter(FloatParameter):
    """
    A parameter that allows to configure the shrinkage parameter, a.k.a. the learning rate, to be used.
    """

    def __init__(self):
        super().__init__(
            name='shrinkage',
            description='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].',
            mixin=ConstantShrinkageMixin)

    def _configure(config, value: float):
        if value == 1.0 and issubclass(type(config), NoPostProcessorMixin):
            config.use_no_post_processor()
        else:
            config.use_constant_shrinkage_post_processor().set_shrinkage(value)


class L1RegularizationParameter(FloatParameter):
    """
    A parameter that allows to configure the weight of the L1 regularization.
    """

    def __init__(self):
        super().__init__(name='l1_regularization_weight',
                         description='The weight of the L1 regularization. Must be at least 0',
                         mixin=L1RegularizationMixin)
        
    def _configure(config, value: float):
        if value == 0.0 and issubclass(type(config), NoL1RegularizationMixin):
            config.use_no_l1_regularization()
        else:
            config.use_l1_regularization().set_regularization_weight(value)


class L2RegularizationParameter(FloatParameter):
    """
    A parameter that allows to configure the weight of the L2 regularization.
    """

    def __init__(self):
        super().__init__(name='l2_regularization_weight',
                         description='The weight of the L2 regularization. Must be at least 0',
                         mixin=L2RegularizationMixin)
        
    def _configure(config, value: float):
        if value == 0.0 and issubclass(type(config), NoL2RegularizationMixin):
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(value)


class DefaultRuleParameter(NominalParameter):
    """
    A parameter that allows to configure whether a default rule should be induced or not.
    """

    def __init__(self):
        super().__init__(name=PARAM_DEFAULT_RULE, description='Whether a default rule should be induced or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoDefaultRuleMixin)
        self.add_value(name=BooleanOption.TRUE.value, mixin=DefaultRuleMixin)
        self.add_value(name=AUTOMATIC, mixin=AutomaticDefaultRuleMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_default_rule()
        elif value == BooleanOption.TRUE.value:
            config.use_default_rule()
        elif value == AUTOMATIC:
            config.use_automatic_default_rule()


class StatisticFormatParameter(NominalParameter):
    """
    A parameter that allows to configure the format to be used for the representation of gradients and Hessians.
    """

    STATISTIC_FORMAT_DENSE = 'dense'

    STATISTIC_FORMAT_SPARSE = 'sparse'

    def __init__(self):
        super().__init__(name='statistic_format',
                         description='The format to be used for the representation of gradients and Hessians')
        self.add_value(name=self.STATISTIC_FORMAT_DENSE, mixin=DenseStatisticsMixin)
        self.add_value(name=self.STATISTIC_FORMAT_SPARSE, mixin=SparseStatisticsMixin)
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticStatisticsMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable format is chosen automatically '
                       + 'based on the parameters ' + PARAM_LOSS + ', ' + PARAM_HEAD_TYPE + ', ' + PARAM_DEFAULT_RULE
                       + ' and the characteristics of the label matrix')

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == STATISTIC_FORMAT_DENSE:
            config.use_dense_statistics()
        elif value == STATISTIC_FORMAT_SPARSE:
            config.use_sparse_statistics()
        elif value == AUTOMATIC:
            config.use_automatic_statistics()


class LabelBinningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for gradient-based label binning (GBLB).
    """

    def __init__(self):
        super().__init__(name='label_binning',
                         description='The name of the strategy to be used for gradient-based label binning (GBLB)')
        self.add_value(name=NONE, mixin=NoLabelBinningMixin)
        self.add_value(name=BINNING_EQUAL_WIDTH,
                       mixin=EqualWidthLabelBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticLabelBinningMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameters ' + PARAM_LOSS + ' and ' + PARAM_HEAD_TYPE)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_label_binning()
        elif value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_label_binning()
            c.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(OPTION_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(OPTION_MAX_BINS, c.get_max_bins()))
        elif value == AUTOMATIC:
            config.use_automatic_label_binning()


class LossParameter(NominalParameter):
    """
    A parameter that allows to configure the loss function to be minimized during training.
    """

    LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

    LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

    LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

    LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

    LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

    LOSS_SQUARED_HINGE_EXAMPLE_WISE = 'squared-hinge-example-wise'

    def __init__(self):
        super().__init__(name=PARAM_LOSS, description='The name of the loss function to be minimized during training')
        self.add_value(name=self.LOSS_LOGISTIC_LABEL_WISE, mixin=LabelWiseLogisticLossMixin)
        self.add_value(name=self.LOSS_LOGISTIC_EXAMPLE_WISE, mixin=ExampleWiseLogisticLossMixin)
        self.add_value(name=self.LOSS_SQUARED_ERROR_LABEL_WISE, mixin=LabelWiseSquaredErrorLossMixin)
        self.add_value(name=self.LOSS_SQUARED_ERROR_EXAMPLE_WISE, mixin=ExampleWiseSquaredErrorLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_LABEL_WISE, mixin=LabelWiseSquaredHingeLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_EXAMPLE_WISE, mixin=ExampleWiseSquaredHingeLossMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.LOSS_LOGISTIC_LABEL_WISE:
            config.use_label_wise_logistic_loss()
        elif value == self.LOSS_LOGISTIC_EXAMPLE_WISE:
            config.use_example_wise_logistic_loss()
        elif value == self.LOSS_SQUARED_ERROR_LABEL_WISE:
            config.use_label_wise_squared_error_loss()
        elif value == self.LOSS_SQUARED_ERROR_EXAMPLE_WISE:
            config.use_example_wise_squared_error_loss()
        elif value == self.LOSS_SQUARED_HINGE_LABEL_WISE:
            config.use_label_wise_squared_hinge_loss()
        elif value == self.LOSS_SQUARED_HINGE_EXAMPLE_WISE:
            config.use_example_wise_squared_hinge_loss()


class HeadTypeParameter(NominalParameter):
    """
    A parameter that allows to configure the type of the rule heads that should be used.
    """

    HEAD_TYPE_SINGLE = 'single-label'

    HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

    OPTION_LABEL_RATIO = 'label_ratio'

    OPTION_MIN_LABELS = 'min_labels'

    OPTION_MAX_LABELS = 'max_labels'

    HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

    OPTION_THRESHOLD = 'threshold'

    OPTION_EXPONENT = 'exponent'

    HEAD_TYPE_COMPLETE = 'complete'

    def __init__(self):
        super().__init__(name=PARAM_HEAD_TYPE, description='The type of the rule heads that should be used')
        self.add_value(name=self.HEAD_TYPE_SINGLE, mixin=SingleLabelHeadMixin)
        self.add_value(name=self.HEAD_TYPE_PARTIAL_FIXED,
                       mixin=FixedPartialHeadMixin,
                       options={self.OPTION_LABEL_RATIO, self.OPTION_MIN_LABELS, self.OPTION_MAX_LABELS})
        self.add_value(name=self.HEAD_TYPE_PARTIAL_DYNAMIC,
                       mixin=DynamicPartialHeadMixin,
                       options={self.OPTION_THRESHOLD, self.OPTION_EXPONENT})
        self.add_value(name=self.HEAD_TYPE_COMPLETE, mixin=CompleteHeadMixin)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == self.HEAD_TYPE_PARTIAL_FIXED:
            c = config.use_fixed_partial_heads()
            c.set_label_ratio(options.get_float(self.OPTION_LABEL_RATIO, c.get_label_ratio()))
            c.set_min_labels(options.get_int(self.OPTION_MIN_LABELS, c.get_min_labels()))
            c.set_max_labels(options.get_int(self.OPTION_MAX_LABELS, c.get_max_labels()))
        elif value == self.HEAD_TYPE_PARTIAL_DYNAMIC:
            c = config.use_dynamic_partial_heads()
            c.set_threshold(options.get_float(self.OPTION_THRESHOLD, c.get_threshold()))
            c.set_exponent(options.get_float(self.OPTION_EXPONENT, c.get_exponent()))
        elif value == self.HEAD_TYPE_COMPLETE:
            config.use_complete_heads()
        elif value == AUTOMATIC:
            config.use_automatic_heads()


class BinaryPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting binary labels.
    """

    BINARY_PREDICTOR_LABEL_WISE = 'label-wise'

    BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

    BINARY_PREDICTOR_GFM = 'gfm'

    def __init__(self):
        super().__init__(name='binary_predictor',
                         description='The name of the strategy to be used for predicting binary labels')
        self.add_value(name=self.BINARY_PREDICTOR_LABEL_WISE, mixin=LabelWiseBinaryPredictorMixin)
        self.add_value(name=self.BINARY_PREDICTOR_EXAMPLE_WISE, mixin=ExampleWiseBinaryPredictorMixin)
        self.add_value(name=self.BINARY_PREDICTOR_GFM, mixin=GfmBinaryPredictorMixin)
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticBinaryPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + PARAM_LOSS)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.BINARY_PREDICTOR_LABEL_WISE:
            config.use_label_wise_binary_predictor()
        elif value == self.BINARY_PREDICTOR_EXAMPLE_WISE:
            config.use_example_wise_binary_predictor()
        elif value == self.BINARY_PREDICTOR_GFM:
            config.use_gfm_binary_predictor()
        elif value == AUTOMATIC:
            config.use_automatic_binary_predictor()


class ProbabilityPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting probabilities.
    """

    PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

    PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

    def __init__(self):
        super().__init__(name='probability_predictor',
                         description='The name of the strategy to be used for predicting probabilities')
        self.add_value(name=self.PROBABILITY_PREDICTOR_LABEL_WISE, mixin=LabelWiseProbabilityPredictorMixin)
        self.add_value(name=self.PROBABILITY_PREDICTOR_MARGINALIZED, mixin=MarginalizedProbabilityPredictorMixin)
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticProbabilityPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + PARAM_LOSS)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.PROBABILITY_PREDICTOR_LABEL_WISE:
            config.use_label_wise_probability_predictor()
        elif value == self.PROBABILITY_PREDICTOR_MARGINALIZED:
            config.use_marginalized_probability_predictor()
        elif value == AUTOMATIC:
            config.use_automatic_probability_predictor()


BOOSTING_RULE_LEARNER_PARAMETERS = RULE_LEARNER_PARAMETERS | {
    ExtendedPartitionSamplingParameter(),
    ExtendedFeatureBinningParameter(),
    ExtendedParallelRuleRefinementParameter(),
    ExtendedParallelStatisticUpdateParameter(),
    ShrinkageParameter(),
    L1RegularizationParameter(),
    L2RegularizationParameter(),
    DefaultRuleParameter(),
    StatisticFormatParameter(),
    LabelBinningParameter(),
    LossParameter(),
    HeadTypeParameter(),
    BinaryPredictorParameter(),
    ProbabilityPredictorParameter()
}
