"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv

from mlrl.common.cython.feature_binning cimport EqualFrequencyFeatureBinningConfig, EqualWidthFeatureBinningConfig, \
    IEqualFrequencyFeatureBinningConfig, IEqualWidthFeatureBinningConfig
from mlrl.common.cython.feature_sampling cimport FeatureSamplingWithoutReplacementConfig, \
    IFeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport ExampleWiseStratifiedInstanceSamplingConfig, \
    IExampleWiseStratifiedInstanceSamplingConfig, IInstanceSamplingWithoutReplacementConfig, \
    IInstanceSamplingWithReplacementConfig, InstanceSamplingWithoutReplacementConfig, \
    InstanceSamplingWithReplacementConfig, IOutputWiseStratifiedInstanceSamplingConfig, \
    OutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig, ManualMultiThreadingConfig
from mlrl.common.cython.output_sampling cimport IOutputSamplingWithoutReplacementConfig, \
    OutputSamplingWithoutReplacementConfig
from mlrl.common.cython.partition_sampling cimport ExampleWiseStratifiedBiPartitionSamplingConfig, \
    IExampleWiseStratifiedBiPartitionSamplingConfig, IOutputWiseStratifiedBiPartitionSamplingConfig, \
    IRandomBiPartitionSamplingConfig, OutputWiseStratifiedBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization cimport ISequentialPostOptimizationConfig, SequentialPostOptimizationConfig
from mlrl.common.cython.rule_induction cimport BeamSearchTopDownRuleInductionConfig, GreedyTopDownRuleInductionConfig, \
    IBeamSearchTopDownRuleInductionConfig, IGreedyTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion cimport IPostPruningConfig, IPrePruningConfig, \
    ISizeStoppingCriterionConfig, ITimeStoppingCriterionConfig, PostPruningConfig, PrePruningConfig, \
    SizeStoppingCriterionConfig, TimeStoppingCriterionConfig

from mlrl.boosting.cython.head_type cimport DynamicPartialHeadConfig, FixedPartialHeadConfig, \
    IDynamicPartialHeadConfig, IFixedPartialHeadConfig
from mlrl.boosting.cython.label_binning cimport EqualWidthLabelBinningConfig, IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport ConstantShrinkageConfig, IConstantShrinkageConfig
from mlrl.boosting.cython.prediction cimport ExampleWiseBinaryPredictorConfig, GfmBinaryPredictorConfig, \
    IExampleWiseBinaryPredictorConfig, IGfmBinaryPredictorConfig, IMarginalizedProbabilityPredictorConfig, \
    IOutputWiseBinaryPredictorConfig, IOutputWiseProbabilityPredictorConfig, MarginalizedProbabilityPredictorConfig, \
    OutputWiseBinaryPredictorConfig, OutputWiseProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration cimport IIsotonicJointProbabilityCalibratorConfig, \
    IIsotonicMarginalProbabilityCalibratorConfig, IsotonicJointProbabilityCalibratorConfig, \
    IsotonicMarginalProbabilityCalibratorConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig, ManualRegularizationConfig

from mlrl.common.cython.learner import BeamSearchTopDownRuleInductionMixin, DefaultRuleMixin, \
    EqualFrequencyFeatureBinningMixin, EqualWidthFeatureBinningMixin, FeatureSamplingWithoutReplacementMixin, \
    GreedyTopDownRuleInductionMixin, InstanceSamplingWithoutReplacementMixin, InstanceSamplingWithReplacementMixin, \
    IrepRulePruningMixin, NoFeatureBinningMixin, NoFeatureSamplingMixin, NoGlobalPruningMixin, \
    NoInstanceSamplingMixin, NoJointProbabilityCalibrationMixin, NoMarginalProbabilityCalibrationMixin, \
    NoOutputSamplingMixin, NoParallelPredictionMixin, NoParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    NoPartitionSamplingMixin, NoPostProcessorMixin, NoRulePruningMixin, NoSequentialPostOptimizationMixin, \
    NoSizeStoppingCriterionMixin, NoTimeStoppingCriterionMixin, OutputSamplingWithoutReplacementMixin, \
    ParallelPredictionMixin, ParallelRuleRefinementMixin, ParallelStatisticUpdateMixin, PostPruningMixin, \
    PrePruningMixin, RandomBiPartitionSamplingMixin, RoundRobinOutputSamplingMixin, SequentialPostOptimizationMixin, \
    SequentialRuleModelAssemblageMixin, SizeStoppingCriterionMixin, TimeStoppingCriterionMixin
from mlrl.common.cython.learner_classification import ExampleWiseStratifiedBiPartitionSamplingMixin, \
    ExampleWiseStratifiedInstanceSamplingMixin, OutputWiseStratifiedBiPartitionSamplingMixin, \
    OutputWiseStratifiedInstanceSamplingMixin

from mlrl.boosting.cython.learner import AutomaticBinaryPredictorMixin, AutomaticDefaultRuleMixin, \
    AutomaticFeatureBinningMixin, AutomaticHeadMixin, AutomaticLabelBinningMixin, \
    AutomaticParallelRuleRefinementMixin, AutomaticParallelStatisticUpdateMixin, AutomaticPartitionSamplingMixin, \
    AutomaticProbabilityPredictorMixin, AutomaticStatisticsMixin, CompleteHeadMixin, ConstantShrinkageMixin, \
    DecomposableLogisticLossMixin, DecomposableSquaredErrorLossMixin, DecomposableSquaredHingeLossMixin, \
    DenseStatisticsMixin, DynamicPartialHeadMixin, EqualWidthLabelBinningMixin, ExampleWiseBinaryPredictorMixin, \
    FixedPartialHeadMixin, GfmBinaryPredictorMixin, IsotonicJointProbabilityCalibrationMixin, \
    IsotonicMarginalProbabilityCalibrationMixin, L1RegularizationMixin, L2RegularizationMixin, \
    MarginalizedProbabilityPredictorMixin, NoDefaultRuleMixin, NoL1RegularizationMixin, NoL2RegularizationMixin, \
    NoLabelBinningMixin, NonDecomposableLogisticLossMixin, NonDecomposableSquaredErrorLossMixin, \
    NonDecomposableSquaredHingeLossMixin, OutputWiseBinaryPredictorMixin, OutputWiseProbabilityPredictorMixin, \
    OutputWiseScorePredictorMixin, SingleOutputHeadMixin, SparseStatisticsMixin


cdef class BoomerConfig(RuleLearnerConfig,
                        AutomaticPartitionSamplingMixin,
                        AutomaticFeatureBinningMixin,
                        AutomaticParallelRuleRefinementMixin,
                        AutomaticParallelStatisticUpdateMixin,
                        ConstantShrinkageMixin,
                        NoL1RegularizationMixin,
                        L1RegularizationMixin,
                        NoL2RegularizationMixin,
                        L2RegularizationMixin,
                        NoDefaultRuleMixin,
                        AutomaticDefaultRuleMixin,
                        CompleteHeadMixin,
                        FixedPartialHeadMixin,
                        DynamicPartialHeadMixin,
                        SingleOutputHeadMixin,
                        AutomaticHeadMixin,
                        DenseStatisticsMixin,
                        SparseStatisticsMixin,
                        AutomaticStatisticsMixin,
                        DecomposableLogisticLossMixin,
                        DecomposableSquaredErrorLossMixin,
                        DecomposableSquaredHingeLossMixin,
                        NonDecomposableLogisticLossMixin,
                        NonDecomposableSquaredHingeLossMixin,
                        NonDecomposableSquaredErrorLossMixin,
                        NoLabelBinningMixin,
                        EqualWidthLabelBinningMixin,
                        AutomaticLabelBinningMixin,
                        IsotonicMarginalProbabilityCalibrationMixin,
                        IsotonicJointProbabilityCalibrationMixin,
                        OutputWiseBinaryPredictorMixin,
                        ExampleWiseBinaryPredictorMixin,
                        GfmBinaryPredictorMixin,
                        AutomaticBinaryPredictorMixin,
                        OutputWiseScorePredictorMixin,
                        OutputWiseProbabilityPredictorMixin,
                        MarginalizedProbabilityPredictorMixin,
                        AutomaticProbabilityPredictorMixin,
                        SequentialRuleModelAssemblageMixin,
                        DefaultRuleMixin,
                        GreedyTopDownRuleInductionMixin,
                        BeamSearchTopDownRuleInductionMixin,
                        NoPostProcessorMixin,
                        NoFeatureBinningMixin,
                        EqualWidthFeatureBinningMixin,
                        EqualFrequencyFeatureBinningMixin,
                        NoOutputSamplingMixin,
                        RoundRobinOutputSamplingMixin,
                        OutputSamplingWithoutReplacementMixin,
                        NoInstanceSamplingMixin,
                        InstanceSamplingWithReplacementMixin,
                        InstanceSamplingWithoutReplacementMixin,
                        OutputWiseStratifiedInstanceSamplingMixin,
                        ExampleWiseStratifiedInstanceSamplingMixin,
                        NoFeatureSamplingMixin,
                        FeatureSamplingWithoutReplacementMixin,
                        NoPartitionSamplingMixin,
                        RandomBiPartitionSamplingMixin,
                        OutputWiseStratifiedBiPartitionSamplingMixin,
                        ExampleWiseStratifiedBiPartitionSamplingMixin,
                        NoRulePruningMixin,
                        IrepRulePruningMixin,
                        NoParallelRuleRefinementMixin,
                        ParallelRuleRefinementMixin,
                        NoParallelStatisticUpdateMixin,
                        ParallelStatisticUpdateMixin,
                        NoParallelPredictionMixin,
                        ParallelPredictionMixin,
                        NoSizeStoppingCriterionMixin,
                        SizeStoppingCriterionMixin,
                        NoTimeStoppingCriterionMixin,
                        TimeStoppingCriterionMixin,
                        PrePruningMixin,
                        NoGlobalPruningMixin,
                        PostPruningMixin,
                        NoSequentialPostOptimizationMixin,
                        SequentialPostOptimizationMixin,
                        NoMarginalProbabilityCalibrationMixin,
                        NoJointProbabilityCalibrationMixin):
    """
    Allows to configure the BOOMER algorithm.
    """

    def __cinit__(self):
        self.config_ptr = createBoomerConfig()

    def use_sequential_rule_model_assemblage(self):
        self.config_ptr.get().useSequentialRuleModelAssemblage()

    def use_default_rule(self):
        self.config_ptr.get().useDefaultRule()

    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        cdef IGreedyTopDownRuleInductionConfig* config_ptr = &self.config_ptr.get().useGreedyTopDownRuleInduction()
        cdef GreedyTopDownRuleInductionConfig config = \
            GreedyTopDownRuleInductionConfig.__new__(GreedyTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        cdef IBeamSearchTopDownRuleInductionConfig* config_ptr = \
            &self.config_ptr.get().useBeamSearchTopDownRuleInduction()
        cdef BeamSearchTopDownRuleInductionConfig config = \
            BeamSearchTopDownRuleInductionConfig.__new__(BeamSearchTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_post_processor(self):
        self.config_ptr.get().useNoPostProcessor()

    def use_no_feature_binning(self):
        self.config_ptr.get().useNoFeatureBinning()

    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        cdef IEqualWidthFeatureBinningConfig* config_ptr = \
            &self.config_ptr.get().useEqualWidthFeatureBinning()
        cdef EqualWidthFeatureBinningConfig config = \
            EqualWidthFeatureBinningConfig.__new__(EqualWidthFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_equal_frequency_feature_binning(self) -> EqualFrequencyFeatureBinningConfig:
        cdef IEqualFrequencyFeatureBinningConfig* config_ptr = \
            &self.config_ptr.get().useEqualFrequencyFeatureBinning()
        cdef EqualFrequencyFeatureBinningConfig config = \
            EqualFrequencyFeatureBinningConfig.__new__(EqualFrequencyFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_output_sampling(self):
        self.config_ptr.get().useNoOutputSampling()

    def use_round_robin_output_sampling(self):
        self.config_ptr.get().useRoundRobinOutputSampling()

    def use_output_sampling_without_replacement(self) -> OutputSamplingWithoutReplacementConfig:
        cdef IOutputSamplingWithoutReplacementConfig* config_ptr = \
            &self.config_ptr.get().useOutputSamplingWithoutReplacement()
        cdef OutputSamplingWithoutReplacementConfig config = \
            OutputSamplingWithoutReplacementConfig.__new__(OutputSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        self.config_ptr.get().useNoInstanceSampling()
    
    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        cdef IInstanceSamplingWithReplacementConfig* config_ptr = \
            &self.config_ptr.get().useInstanceSamplingWithReplacement()
        cdef InstanceSamplingWithReplacementConfig config = \
            InstanceSamplingWithReplacementConfig.__new__(InstanceSamplingWithReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        cdef IInstanceSamplingWithoutReplacementConfig* config_ptr = \
            &self.config_ptr.get().useInstanceSamplingWithoutReplacement()
        cdef InstanceSamplingWithoutReplacementConfig config = \
            InstanceSamplingWithoutReplacementConfig.__new__(InstanceSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_output_wise_stratified_instance_sampling(self) -> OutputWiseStratifiedInstanceSamplingConfig:
        cdef IOutputWiseStratifiedInstanceSamplingConfig* config_ptr = \
            &self.config_ptr.get().useOutputWiseStratifiedInstanceSampling()
        cdef OutputWiseStratifiedInstanceSamplingConfig config = \
            OutputWiseStratifiedInstanceSamplingConfig.__new__(OutputWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr = \
            &self.config_ptr.get().useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = \
            ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        self.config_ptr.get().useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        cdef IFeatureSamplingWithoutReplacementConfig* config_ptr = \
            &self.config_ptr.get().useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = \
            FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        self.config_ptr.get().useNoPartitionSampling()
    
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        cdef IRandomBiPartitionSamplingConfig* config_ptr = \
            &self.config_ptr.get().useRandomBiPartitionSampling()
        cdef RandomBiPartitionSamplingConfig config = \
            RandomBiPartitionSamplingConfig.__new__(RandomBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_output_wise_stratified_bi_partition_sampling(self) -> OutputWiseStratifiedBiPartitionSamplingConfig:
        cdef IOutputWiseStratifiedBiPartitionSamplingConfig* config_ptr = \
            &self.config_ptr.get().useOutputWiseStratifiedBiPartitionSampling()
        cdef OutputWiseStratifiedBiPartitionSamplingConfig config = \
            OutputWiseStratifiedBiPartitionSamplingConfig.__new__(OutputWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = \
            &self.config_ptr.get().useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = \
            ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_partition_sampling(self):
        self.config_ptr.get().useAutomaticPartitionSampling()

    def use_no_rule_pruning(self):
        self.config_ptr.get().useNoRulePruning()

    def use_irep_rule_pruning(self):
        self.config_ptr.get().useIrepRulePruning()

    def use_no_parallel_rule_refinement(self):
        self.config_ptr.get().useNoParallelRuleRefinement()
        
    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        cdef IManualMultiThreadingConfig* config_ptr = &self.config_ptr.get().useParallelRuleRefinement()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_statistic_update(self):
        self.config_ptr.get().useNoParallelStatisticUpdate()

    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        cdef IManualMultiThreadingConfig* config_ptr = &self.config_ptr.get().useParallelStatisticUpdate()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_prediction(self):
        self.config_ptr.get().useNoParallelPrediction()

    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        cdef IManualMultiThreadingConfig* config_ptr = &self.config_ptr.get().useParallelPrediction()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_size_stopping_criterion(self):
        self.config_ptr.get().useNoSizeStoppingCriterion()

    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        cdef ISizeStoppingCriterionConfig* config_ptr = &self.config_ptr.get().useSizeStoppingCriterion()
        cdef SizeStoppingCriterionConfig config = SizeStoppingCriterionConfig.__new__(SizeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_time_stopping_criterion(self):
        self.config_ptr.get().useNoTimeStoppingCriterion()

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        cdef ITimeStoppingCriterionConfig* config_ptr = &self.config_ptr.get().useTimeStoppingCriterion()
        cdef TimeStoppingCriterionConfig config = TimeStoppingCriterionConfig.__new__(TimeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_global_pre_pruning(self) -> PrePruningConfig:
        cdef IPrePruningConfig* config_ptr = &self.config_ptr.get().useGlobalPrePruning()
        cdef PrePruningConfig config = PrePruningConfig.__new__(PrePruningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_global_pruning(self):
        self.config_ptr.get().useNoGlobalPruning()

    def use_global_post_pruning(self) -> PostPruningConfig:
        cdef IPostPruningConfig* config_ptr = &self.config_ptr.get().useGlobalPostPruning()
        cdef PostPruningConfig config = PostPruningConfig.__new__(PostPruningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_sequential_post_optimization(self):
        self.config_ptr.get().useNoSequentialPostOptimization()

    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        cdef ISequentialPostOptimizationConfig* config_ptr = &self.config_ptr.get().useSequentialPostOptimization()
        cdef SequentialPostOptimizationConfig config = \
            SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_default_rule(self):
        self.config_ptr.get().useNoDefaultRule()

    def use_automatic_default_rule(self):
        self.config_ptr.get().useAutomaticDefaultRule()

    def use_automatic_feature_binning(self):
        self.config_ptr.get().useAutomaticFeatureBinning()

    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        cdef IConstantShrinkageConfig* config_ptr = &self.config_ptr.get().useConstantShrinkagePostProcessor()
        cdef ConstantShrinkageConfig config = ConstantShrinkageConfig.__new__(ConstantShrinkageConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_parallel_rule_refinement(self):
        self.config_ptr.get().useAutomaticParallelRuleRefinement()

    def use_automatic_parallel_statistic_update(self):
        self.config_ptr.get().useAutomaticParallelStatisticUpdate()

    def use_automatic_heads(self):
        self.config_ptr.get().useAutomaticHeads()

    def use_complete_heads(self):
        self.config_ptr.get().useCompleteHeads()

    def use_fixed_partial_heads(self) -> FixedPartialHeadConfig:
        cdef IFixedPartialHeadConfig* config_ptr = &self.config_ptr.get().useFixedPartialHeads()
        cdef FixedPartialHeadConfig config = FixedPartialHeadConfig.__new__(FixedPartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_dynamic_partial_heads(self) -> DynamicPartialHeadConfig:
        cdef IDynamicPartialHeadConfig* config_ptr = &self.config_ptr.get().useDynamicPartialHeads()
        cdef DynamicPartialHeadConfig config = DynamicPartialHeadConfig.__new__(DynamicPartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_output_heads(self):
        self.config_ptr.get().useSingleOutputHeads()

    def use_automatic_statistics(self):
        self.config_ptr.get().useAutomaticStatistics()

    def use_dense_statistics(self):
        self.config_ptr.get().useDenseStatistics()

    def use_sparse_statistics(self):
        self.config_ptr.get().useSparseStatistics()

    def use_no_l1_regularization(self):
        self.config_ptr.get().useNoL1Regularization()

    def use_l1_regularization(self) -> ManualRegularizationConfig:
        cdef IManualRegularizationConfig* config_ptr = &self.config_ptr.get().useL1Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_l2_regularization(self):
        self.config_ptr.get().useNoL2Regularization()

    def use_l2_regularization(self) -> ManualRegularizationConfig:
        cdef IManualRegularizationConfig* config_ptr = &self.config_ptr.get().useL2Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_non_decomposable_logistic_loss(self):
        self.config_ptr.get().useNonDecomposableLogisticLoss()

    def use_non_decomposable_squared_error_loss(self):
        self.config_ptr.get().useNonDecomposableSquaredErrorLoss()

    def use_non_decomposable_squared_hinge_loss(self):
        self.config_ptr.get().useNonDecomposableSquaredHingeLoss()

    def use_decomposable_logistic_loss(self):
        self.config_ptr.get().useDecomposableLogisticLoss()

    def use_decomposable_squared_error_loss(self):
        self.config_ptr.get().useDecomposableSquaredErrorLoss()

    def use_decomposable_squared_hinge_loss(self):
        self.config_ptr.get().useDecomposableSquaredHingeLoss()

    def use_no_label_binning(self):
        self.config_ptr.get().useNoLabelBinning()

    def use_automatic_label_binning(self):
        self.config_ptr.get().useAutomaticLabelBinning()

    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        cdef IEqualWidthLabelBinningConfig* config_ptr = &self.config_ptr.get().useEqualWidthLabelBinning()
        cdef EqualWidthLabelBinningConfig config = EqualWidthLabelBinningConfig.__new__(EqualWidthLabelBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_marginal_probability_calibration(self):
        self.config_ptr.get().useNoMarginalProbabilityCalibration()

    def use_isotonic_marginal_probability_calibration(self) -> IsotonicMarginalProbabilityCalibratorConfig:
        cdef IIsotonicMarginalProbabilityCalibratorConfig* config_ptr = \
            &self.config_ptr.get().useIsotonicMarginalProbabilityCalibration()
        cdef IsotonicMarginalProbabilityCalibratorConfig config = \
            IsotonicMarginalProbabilityCalibratorConfig.__new__(IsotonicMarginalProbabilityCalibratorConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_joint_probability_calibration(self):
        self.config_ptr.get().useNoJointProbabilityCalibration()

    def use_isotonic_joint_probability_calibration(self) -> IsotonicJointProbabilityCalibratorConfig:
        cdef IIsotonicJointProbabilityCalibratorConfig* config_ptr = \
            &self.config_ptr.get().useIsotonicJointProbabilityCalibration()
        cdef IsotonicJointProbabilityCalibratorConfig config = \
            IsotonicJointProbabilityCalibratorConfig.__new__(IsotonicJointProbabilityCalibratorConfig)
        config.config_ptr = config_ptr
        return config

    def use_output_wise_binary_predictor(self) -> OutputWiseBinaryPredictorConfig:
        cdef IOutputWiseBinaryPredictorConfig* config_ptr = &self.config_ptr.get().useOutputWiseBinaryPredictor()
        cdef OutputWiseBinaryPredictorConfig config = \
            OutputWiseBinaryPredictorConfig.__new__(OutputWiseBinaryPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_binary_predictor(self) -> ExampleWiseBinaryPredictorConfig:
        cdef IExampleWiseBinaryPredictorConfig* config_ptr = &self.config_ptr.get().useExampleWiseBinaryPredictor()
        cdef ExampleWiseBinaryPredictorConfig config = \
            ExampleWiseBinaryPredictorConfig.__new__(ExampleWiseBinaryPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_gfm_binary_predictor(self) -> GfmBinaryPredictorConfig:
        cdef IGfmBinaryPredictorConfig* config_ptr = &self.config_ptr.get().useGfmBinaryPredictor()
        cdef GfmBinaryPredictorConfig config = GfmBinaryPredictorConfig.__new__(GfmBinaryPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_binary_predictor(self):
        self.config_ptr.get().useOutputWiseBinaryPredictor()

    def use_output_wise_score_predictor(self):
        self.config_ptr.get().useOutputWiseScorePredictor()

    def use_output_wise_probability_predictor(self) -> OutputWiseProbabilityPredictorConfig:
        cdef IOutputWiseProbabilityPredictorConfig* config_ptr = \
            &self.config_ptr.get().useOutputWiseProbabilityPredictor()
        cdef OutputWiseProbabilityPredictorConfig config = \
            OutputWiseProbabilityPredictorConfig.__new__(OutputWiseProbabilityPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_marginalized_probability_predictor(self) -> MarginalizedProbabilityPredictorConfig:
        cdef IMarginalizedProbabilityPredictorConfig* config_ptr = \
            &self.config_ptr.get().useMarginalizedProbabilityPredictor()
        cdef MarginalizedProbabilityPredictorConfig config = \
            MarginalizedProbabilityPredictorConfig.__new__(MarginalizedProbabilityPredictorConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_probability_predictor(self):
        self.config_ptr.get().useAutomaticProbabilityPredictor()


cdef class Boomer(RuleLearner):
    """
    The BOOMER rule learning algorithm.
    """

    def __cinit__(self, BoomerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createBoomer(move(config.config_ptr), ddot, dspmv, dsysv)

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
