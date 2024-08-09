"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility function for configuring boosting algorithms.
"""
from typing import Optional

from mlrl.common.config import AUTOMATIC, BINNING_EQUAL_WIDTH, NONE, OPTION_BIN_RATIO, OPTION_MAX_BINS, \
    OPTION_MIN_BINS, OPTION_USE_HOLDOUT_SET, RULE_LEARNER_PARAMETERS, FeatureBinningParameter, FloatParameter, \
    NominalParameter, ParallelRuleRefinementParameter, ParallelStatisticUpdateParameter, PartitionSamplingParameter
from mlrl.common.cython.learner import DefaultRuleMixin, NoJointProbabilityCalibrationMixin, \
    NoMarginalProbabilityCalibrationMixin, NoPostProcessorMixin
from mlrl.common.options import BooleanOption, Options

from mlrl.boosting.cython.learner import AutomaticFeatureBinningMixin, AutomaticParallelRuleRefinementMixin, \
    AutomaticParallelStatisticUpdateMixin, CompleteHeadMixin, ConstantShrinkageMixin, \
    DecomposableSquaredErrorLossMixin, DynamicPartialHeadMixin, FixedPartialHeadMixin, L1RegularizationMixin, \
    L2RegularizationMixin, NoL1RegularizationMixin, NoL2RegularizationMixin, NonDecomposableSquaredErrorLossMixin, \
    SingleOutputHeadMixin
from mlrl.boosting.cython.learner_classification import AutomaticBinaryPredictorMixin, AutomaticDefaultRuleMixin, \
    AutomaticLabelBinningMixin, AutomaticPartitionSamplingMixin, AutomaticProbabilityPredictorMixin, \
    AutomaticStatisticsMixin, DecomposableLogisticLossMixin, DecomposableSquaredHingeLossMixin, DenseStatisticsMixin, \
    EqualWidthLabelBinningMixin, ExampleWiseBinaryPredictorMixin, GfmBinaryPredictorMixin, \
    IsotonicJointProbabilityCalibrationMixin, IsotonicMarginalProbabilityCalibrationMixin, \
    MarginalizedProbabilityPredictorMixin, NoDefaultRuleMixin, NoLabelBinningMixin, NonDecomposableLogisticLossMixin, \
    NonDecomposableSquaredHingeLossMixin, OutputWiseBinaryPredictorMixin, OutputWiseProbabilityPredictorMixin, \
    SparseStatisticsMixin

PROBABILITY_CALIBRATION_ISOTONIC = 'isotonic'

OPTION_BASED_ON_PROBABILITIES = 'based_on_probabilities'

OPTION_USE_PROBABILITY_CALIBRATION_MODEL = 'use_probability_calibration'


class ExtendedPartitionSamplingParameter(PartitionSamplingParameter):
    """
    Extends the `PartitionSamplingParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticPartitionSamplingMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'depending on whether a holdout set is needed and depending on the loss function')

    def _configure(self, config, value: str, options: Optional[Options]):
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
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticFeatureBinningMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the characteristics of the feature matrix')

    def _configure(self, config, value: str, options: Optional[Options]):
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
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticParallelRuleRefinementMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on ' + 'the parameter ' + RegressionLossParameter().argument_name)

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
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticParallelStatisticUpdateMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on ' + 'the parameter ' + RegressionLossParameter().argument_name)

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

    def _configure(self, config, value):
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

    def _configure(self, config, value):
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

    def _configure(self, config, value):
        if value == 0.0 and issubclass(type(config), NoL2RegularizationMixin):
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(value)


class DefaultRuleParameter(NominalParameter):
    """
    A parameter that allows to configure whether a default rule should be induced or not.
    """

    def __init__(self):
        super().__init__(name='default_rule', description='Whether a default rule should be induced or not')
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
                       + 'based on the parameters ' + RegressionLossParameter().argument_name + ', '
                       + HeadTypeParameter().argument_name + ', ' + DefaultRuleParameter().argument_name + ' and the '
                       + 'characteristics of the label matrix')

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.STATISTIC_FORMAT_DENSE:
            config.use_dense_statistics()
        elif value == self.STATISTIC_FORMAT_SPARSE:
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
                       + 'based on the parameters ' + RegressionLossParameter().argument_name + ' and '
                       + HeadTypeParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_label_binning()
        elif value == BINNING_EQUAL_WIDTH:
            conf = config.use_equal_width_label_binning()
            conf.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, conf.get_bin_ratio()))
            conf.set_min_bins(options.get_int(OPTION_MIN_BINS, conf.get_min_bins()))
            conf.set_max_bins(options.get_int(OPTION_MAX_BINS, conf.get_max_bins()))
        elif value == AUTOMATIC:
            config.use_automatic_label_binning()


class RegressionLossParameter(NominalParameter):
    """
    A parameter that allows to configure the loss function to be minimized during training in regression problems.
    """

    LOSS_SQUARED_ERROR_DECOMPOSABLE = 'squared-error-decomposable'

    LOSS_SQUARED_ERROR_NON_DECOMPOSABLE = 'squared-error-non-decomposable'

    def __init__(self):
        super().__init__(name='loss', description='The name of the loss function to be minimized during training')
        self.add_value(name=self.LOSS_SQUARED_ERROR_DECOMPOSABLE, mixin=DecomposableSquaredErrorLossMixin)
        self.add_value(name=self.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE, mixin=NonDecomposableSquaredErrorLossMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.LOSS_SQUARED_ERROR_DECOMPOSABLE:
            config.use_decomposable_squared_error_loss()
        elif value == self.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE:
            config.use_non_decomposable_squared_error_loss()


class ClassificationLossParameter(RegressionLossParameter):
    """
    A parameter that allows to configure the loss function to be minimized during training in classification problems.
    """

    LOSS_LOGISTIC_DECOMPOSABLE = 'logistic-decomposable'

    LOSS_LOGISTIC_NON_DECOMPOSABLE = 'logistic-non-decomposable'

    LOSS_SQUARED_HINGE_DECOMPOSABLE = 'squared-hinge-decomposable'

    LOSS_SQUARED_HINGE_NON_DECOMPOSABLE = 'squared-hinge-non-decomposable'

    def __init__(self):
        super().__init__()
        self.add_value(name=self.LOSS_LOGISTIC_DECOMPOSABLE, mixin=DecomposableLogisticLossMixin)
        self.add_value(name=self.LOSS_LOGISTIC_NON_DECOMPOSABLE, mixin=NonDecomposableLogisticLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_DECOMPOSABLE, mixin=DecomposableSquaredHingeLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE, mixin=NonDecomposableSquaredHingeLossMixin)

    def _configure(self, config, value: str, options: Optional[Options]):
        super()._configure(config, value, options)

        if value == self.LOSS_LOGISTIC_DECOMPOSABLE:
            config.use_decomposable_logistic_loss()
        elif value == self.LOSS_LOGISTIC_NON_DECOMPOSABLE:
            config.use_non_decomposable_logistic_loss()
        elif value == self.LOSS_SQUARED_HINGE_DECOMPOSABLE:
            config.use_decomposable_squared_hinge_loss()
        elif value == self.LOSS_SQUARED_HINGE_NON_DECOMPOSABLE:
            config.use_non_decomposable_squared_hinge_loss()


class HeadTypeParameter(NominalParameter):
    """
    A parameter that allows to configure the type of the rule heads that should be used.
    """

    HEAD_TYPE_SINGLE = 'single'

    HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

    OPTION_OUTPUT_RATIO = 'output_ratio'

    OPTION_MIN_OUTPUTS = 'min_outputs'

    OPTION_MAX_OUTPUTS = 'max_outputs'

    HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

    OPTION_THRESHOLD = 'threshold'

    OPTION_EXPONENT = 'exponent'

    HEAD_TYPE_COMPLETE = 'complete'

    def __init__(self):
        super().__init__(name='head_type', description='The type of the rule heads that should be used')
        self.add_value(name=self.HEAD_TYPE_SINGLE, mixin=SingleOutputHeadMixin)
        self.add_value(name=self.HEAD_TYPE_PARTIAL_FIXED,
                       mixin=FixedPartialHeadMixin,
                       options={self.OPTION_OUTPUT_RATIO, self.OPTION_MIN_OUTPUTS, self.OPTION_MAX_OUTPUTS})
        self.add_value(name=self.HEAD_TYPE_PARTIAL_DYNAMIC,
                       mixin=DynamicPartialHeadMixin,
                       options={self.OPTION_THRESHOLD, self.OPTION_EXPONENT})
        self.add_value(name=self.HEAD_TYPE_COMPLETE, mixin=CompleteHeadMixin)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.HEAD_TYPE_SINGLE:
            config.use_single_output_heads()
        elif value == self.HEAD_TYPE_PARTIAL_FIXED:
            conf = config.use_fixed_partial_heads()
            conf.set_output_ratio(options.get_float(self.OPTION_OUTPUT_RATIO, conf.get_output_ratio()))
            conf.set_min_outputs(options.get_int(self.OPTION_MIN_OUTPUTS, conf.get_min_outputs()))
            conf.set_max_outputs(options.get_int(self.OPTION_MAX_OUTPUTS, conf.get_max_outputs()))
        elif value == self.HEAD_TYPE_PARTIAL_DYNAMIC:
            conf = config.use_dynamic_partial_heads()
            conf.set_threshold(options.get_float(self.OPTION_THRESHOLD, conf.get_threshold()))
            conf.set_exponent(options.get_float(self.OPTION_EXPONENT, conf.get_exponent()))
        elif value == self.HEAD_TYPE_COMPLETE:
            config.use_complete_heads()
        elif value == AUTOMATIC:
            config.use_automatic_heads()


class MarginalProbabilityCalibrationParameter(NominalParameter):
    """
    A parameter that allows to configure the method to be used for the calibration of marginal probabilities.
    """

    def __init__(self):
        super().__init__(name='marginal_probability_calibration',
                         description='The name of the method to be used for the calibration of marginal probabilities')
        self.add_value(name=NONE, mixin=NoMarginalProbabilityCalibrationMixin)
        self.add_value(name=PROBABILITY_CALIBRATION_ISOTONIC,
                       mixin=IsotonicMarginalProbabilityCalibrationMixin,
                       options={OPTION_USE_HOLDOUT_SET})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_marginal_probability_calibration()
        if value == PROBABILITY_CALIBRATION_ISOTONIC:
            conf = config.use_isotonic_marginal_probability_calibration()
            conf.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, conf.is_holdout_set_used()))


class JointProbabilityCalibrationParameter(NominalParameter):
    """
    A parameter that allows to configure the method to be used for the calibration of joint probabilities.
    """

    def __init__(self):
        super().__init__(name='joint_probability_calibration',
                         description='The name of the method to be used for the calibration of joint probabilities')
        self.add_value(name=NONE, mixin=NoJointProbabilityCalibrationMixin)
        self.add_value(name=PROBABILITY_CALIBRATION_ISOTONIC,
                       mixin=IsotonicJointProbabilityCalibrationMixin,
                       options={OPTION_USE_HOLDOUT_SET})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_joint_probability_calibration()
        if value == PROBABILITY_CALIBRATION_ISOTONIC:
            conf = config.use_isotonic_joint_probability_calibration()
            conf.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, conf.is_holdout_set_used()))


class BinaryPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting binary labels.
    """

    BINARY_PREDICTOR_OUTPUT_WISE = 'output-wise'

    BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

    BINARY_PREDICTOR_GFM = 'gfm'

    def __init__(self):
        super().__init__(name='binary_predictor',
                         description='The name of the strategy to be used for predicting binary labels')
        self.add_value(name=self.BINARY_PREDICTOR_OUTPUT_WISE,
                       mixin=OutputWiseBinaryPredictorMixin,
                       options={OPTION_BASED_ON_PROBABILITIES, OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.BINARY_PREDICTOR_EXAMPLE_WISE,
                       mixin=ExampleWiseBinaryPredictorMixin,
                       options={OPTION_BASED_ON_PROBABILITIES, OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.BINARY_PREDICTOR_GFM,
                       mixin=GfmBinaryPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticBinaryPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + RegressionLossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.BINARY_PREDICTOR_OUTPUT_WISE:
            conf = config.use_output_wise_binary_predictor()
            conf.set_based_on_probabilities(
                options.get_bool(OPTION_BASED_ON_PROBABILITIES, conf.is_based_on_probabilities()))
            conf.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL,
                                 conf.is_probability_calibration_model_used()))
        elif value == self.BINARY_PREDICTOR_EXAMPLE_WISE:
            conf = config.use_example_wise_binary_predictor()
            conf.set_based_on_probabilities(
                options.get_bool(OPTION_BASED_ON_PROBABILITIES, conf.is_based_on_probabilities()))
            conf.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL,
                                 conf.is_probability_calibration_model_used()))
        elif value == self.BINARY_PREDICTOR_GFM:
            conf = config.use_gfm_binary_predictor()
            conf.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL,
                                 conf.is_probability_calibration_model_used()))
        elif value == AUTOMATIC:
            config.use_automatic_binary_predictor()


class ProbabilityPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting probabilities.
    """

    PROBABILITY_PREDICTOR_OUTPUT_WISE = 'output-wise'

    PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

    def __init__(self):
        super().__init__(name='probability_predictor',
                         description='The name of the strategy to be used for predicting probabilities')
        self.add_value(name=self.PROBABILITY_PREDICTOR_OUTPUT_WISE,
                       mixin=OutputWiseProbabilityPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.PROBABILITY_PREDICTOR_MARGINALIZED,
                       mixin=MarginalizedProbabilityPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticProbabilityPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + RegressionLossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.PROBABILITY_PREDICTOR_OUTPUT_WISE:
            conf = config.use_output_wise_probability_predictor()
            conf.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL,
                                 conf.is_probability_calibration_model_used()))
        elif value == self.PROBABILITY_PREDICTOR_MARGINALIZED:
            conf = config.use_marginalized_probability_predictor()
            conf.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL,
                                 conf.is_probability_calibration_model_used()))
        elif value == AUTOMATIC:
            config.use_automatic_probability_predictor()


BOOMER_CLASSIFIER_PARAMETERS = RULE_LEARNER_PARAMETERS | {
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
    ClassificationLossParameter(),
    HeadTypeParameter(),
    MarginalProbabilityCalibrationParameter(),
    JointProbabilityCalibrationParameter(),
    BinaryPredictorParameter(),
    ProbabilityPredictorParameter()
}

BOOMER_REGRESSOR_PARAMETERS = RULE_LEARNER_PARAMETERS | {
    ExtendedPartitionSamplingParameter(),
    ExtendedFeatureBinningParameter(),
    ExtendedParallelRuleRefinementParameter(),
    ExtendedParallelStatisticUpdateParameter(),
    ShrinkageParameter(),
    L1RegularizationParameter(),
    L2RegularizationParameter(),
    DefaultRuleParameter(),
    StatisticFormatParameter(),
    RegressionLossParameter(),
    HeadTypeParameter()
}
