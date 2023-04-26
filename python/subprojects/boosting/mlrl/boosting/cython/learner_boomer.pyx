"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.cython.head_type cimport IFixedPartialHeadConfig, FixedPartialHeadConfig, \
    IDynamicPartialHeadConfig, DynamicPartialHeadConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig, EqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig, ConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig, ManualRegularizationConfig
from mlrl.common.cython.feature_binning cimport IEqualWidthFeatureBinningConfig, EqualWidthFeatureBinningConfig, \
    IEqualFrequencyFeatureBinningConfig, EqualFrequencyFeatureBinningConfig
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingWithoutReplacementConfig, \
    FeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport IExampleWiseStratifiedInstanceSamplingConfig, \
    ExampleWiseStratifiedInstanceSamplingConfig, ILabelWiseStratifiedInstanceSamplingConfig, \
    LabelWiseStratifiedInstanceSamplingConfig, IInstanceSamplingWithReplacementConfig, \
    InstanceSamplingWithReplacementConfig, IInstanceSamplingWithoutReplacementConfig, \
    InstanceSamplingWithoutReplacementConfig
from mlrl.common.cython.label_sampling cimport ILabelSamplingWithoutReplacementConfig, \
    LabelSamplingWithoutReplacementConfig
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig, ManualMultiThreadingConfig
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    ExampleWiseStratifiedBiPartitionSamplingConfig, ILabelWiseStratifiedBiPartitionSamplingConfig, \
    LabelWiseStratifiedBiPartitionSamplingConfig, IRandomBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization cimport ISequentialPostOptimizationConfig, SequentialPostOptimizationConfig
from mlrl.common.cython.rule_induction cimport IGreedyTopDownRuleInductionConfig, GreedyTopDownRuleInductionConfig, \
    IBeamSearchTopDownRuleInductionConfig, BeamSearchTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion cimport ISizeStoppingCriterionConfig, SizeStoppingCriterionConfig, \
    ITimeStoppingCriterionConfig, TimeStoppingCriterionConfig, IPrePruningConfig, PrePruningConfig, \
    IPostPruningConfig, PostPruningConfig

from libcpp.utility cimport move

from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv

from mlrl.common.cython.learner import SequentialRuleModelAssemblageMixin, DefaultRuleMixin, \
    GreedyTopDownRuleInductionMixin, BeamSearchTopDownRuleInductionMixin, NoPostProcessorMixin, NoFeatureBinningMixin, \
    EqualWidthFeatureBinningMixin, EqualFrequencyFeatureBinningMixin, NoLabelSamplingMixin, \
    LabelSamplingWithoutReplacementMixin, NoInstanceSamplingMixin, InstanceSamplingWithReplacementMixin, \
    InstanceSamplingWithoutReplacementMixin, LabelWiseStratifiedInstanceSamplingMixin, \
    ExampleWiseStratifiedInstanceSamplingMixin, NoFeatureSamplingMixin, FeatureSamplingWithoutReplacementMixin, \
    NoPartitionSamplingMixin, RandomBiPartitionSamplingMixin, LabelWiseStratifiedBiPartitionSamplingMixin, \
    ExampleWiseStratifiedBiPartitionSamplingMixin, NoRulePruningMixin, IrepRulePruningMixin, \
    NoParallelRuleRefinementMixin, ParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    ParallelStatisticUpdateMixin, NoParallelPredictionMixin, ParallelPredictionMixin, NoSizeStoppingCriterionMixin, \
    SizeStoppingCriterionMixin, NoTimeStoppingCriterionMixin, TimeStoppingCriterionMixin, PrePruningMixin, \
    NoGlobalPruningMixin, PostPruningMixin, NoSequentialPostOptimizationMixin, SequentialPostOptimizationMixin


cdef class BoomerConfig(RuleLearnerConfig,
                        SequentialRuleModelAssemblageMixin,
                        DefaultRuleMixin,
                        GreedyTopDownRuleInductionMixin,
                        BeamSearchTopDownRuleInductionMixin,
                        NoPostProcessorMixin,
                        NoFeatureBinningMixin,
                        EqualWidthFeatureBinningMixin,
                        EqualFrequencyFeatureBinningMixin,
                        NoLabelSamplingMixin,
                        LabelSamplingWithoutReplacementMixin,
                        NoInstanceSamplingMixin,
                        InstanceSamplingWithReplacementMixin,
                        InstanceSamplingWithoutReplacementMixin,
                        LabelWiseStratifiedInstanceSamplingMixin,
                        ExampleWiseStratifiedInstanceSamplingMixin,
                        NoFeatureSamplingMixin,
                        FeatureSamplingWithoutReplacementMixin,
                        NoPartitionSamplingMixin,
                        RandomBiPartitionSamplingMixin,
                        LabelWiseStratifiedBiPartitionSamplingMixin,
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
                        SequentialPostOptimizationMixin):
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
        cdef GreedyTopDownRuleInductionConfig config = GreedyTopDownRuleInductionConfig.__new__(GreedyTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        cdef IBeamSearchTopDownRuleInductionConfig* config_ptr = &self.config_ptr.get().useBeamSearchTopDownRuleInduction()
        cdef BeamSearchTopDownRuleInductionConfig config = BeamSearchTopDownRuleInductionConfig.__new__(BeamSearchTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_post_processor(self):
        self.config_ptr.get().useNoPostProcessor()

    def use_no_feature_binning(self):
        self.config_ptr.get().useNoFeatureBinning()

    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        cdef IEqualWidthFeatureBinningConfig* config_ptr = &self.config_ptr.get().useEqualWidthFeatureBinning()
        cdef EqualWidthFeatureBinningConfig config = EqualWidthFeatureBinningConfig.__new__(EqualWidthFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_equal_frequency_feature_binning(self) -> EqualFrequencyFeatureBinningConfig:
        cdef IEqualFrequencyFeatureBinningConfig* config_ptr = &self.config_ptr.get().useEqualFrequencyFeatureBinning()
        cdef EqualFrequencyFeatureBinningConfig config = EqualFrequencyFeatureBinningConfig.__new__(EqualFrequencyFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_sampling(self):
        self.config_ptr.get().useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        cdef ILabelSamplingWithoutReplacementConfig* config_ptr = &self.config_ptr.get().useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        self.config_ptr.get().useNoInstanceSampling()
    
    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        cdef IInstanceSamplingWithReplacementConfig* config_ptr = &self.config_ptr.get().useInstanceSamplingWithReplacement()
        cdef InstanceSamplingWithReplacementConfig config = InstanceSamplingWithReplacementConfig.__new__(InstanceSamplingWithReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        cdef IInstanceSamplingWithoutReplacementConfig* config_ptr = &self.config_ptr.get().useInstanceSamplingWithoutReplacement()
        cdef InstanceSamplingWithoutReplacementConfig config = InstanceSamplingWithoutReplacementConfig.__new__(InstanceSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_instance_sampling(self) -> LabelWiseStratifiedInstanceSamplingConfig:
        cdef ILabelWiseStratifiedInstanceSamplingConfig* config_ptr = &self.config_ptr.get().useLabelWiseStratifiedInstanceSampling()
        cdef LabelWiseStratifiedInstanceSamplingConfig config = LabelWiseStratifiedInstanceSamplingConfig.__new__(LabelWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr = &self.config_ptr.get().useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        self.config_ptr.get().useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        cdef IFeatureSamplingWithoutReplacementConfig* config_ptr = &self.config_ptr.get().useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        self.config_ptr.get().useNoPartitionSampling()
    
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        cdef IRandomBiPartitionSamplingConfig* config_ptr = &self.config_ptr.get().useRandomBiPartitionSampling()
        cdef RandomBiPartitionSamplingConfig config = RandomBiPartitionSamplingConfig.__new__(RandomBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_bi_partition_sampling(self) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        cdef ILabelWiseStratifiedBiPartitionSamplingConfig* config_ptr = &self.config_ptr.get().useLabelWiseStratifiedBiPartitionSampling()
        cdef LabelWiseStratifiedBiPartitionSamplingConfig config = LabelWiseStratifiedBiPartitionSamplingConfig.__new__(LabelWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = &self.config_ptr.get().useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_partition_sampling(self):
        """
        Configures the rule learner to automatically decide whether a holdout set should be used or not.
        """
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
        cdef SequentialPostOptimizationConfig config = SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_default_rule(self):
        """
        Configures the rule learner to not induce a default rule.
        """
        self.config_ptr.get().useNoDefaultRule()

    def use_automatic_default_rule(self):
        """
        Configures the rule learner to automatically decide whether a default rule should be induced or not.
        """
        self.config_ptr.get().useAutomaticDefaultRule()

    def use_automatic_feature_binning(self):
        """
        Configures the rule learning to automatically decide whether a method for the assignment of numerical feature
        values to bins should be used or not.
        """
        self.config_ptr.get().useAutomaticFeatureBinning()

    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        cdef IConstantShrinkageConfig* config_ptr = &self.config_ptr.get().useConstantShrinkagePostProcessor()
        cdef ConstantShrinkageConfig config = ConstantShrinkageConfig.__new__(ConstantShrinkageConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_parallel_rule_refinement(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        refinement of rules or not.
        """
        self.config_ptr.get().useAutomaticParallelRuleRefinement()

    def use_automatic_parallel_statistic_update(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        update of statistics or not.
        """
        self.config_ptr.get().useAutomaticParallelStatisticUpdate()

    def use_automatic_heads(self):
        """
        Configures the rule learner to automatically decide for the type of rule heads to be used.
        """
        self.config_ptr.get().useAutomaticHeads()

    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available labels.
        """
        self.config_ptr.get().useCompleteHeads()

    def use_fixed_partial_heads(self) -> FixedPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a predefined number of labels.

        :return: A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        cdef IFixedPartialHeadConfig* config_ptr = &self.config_ptr.get().useFixedPartialHeads()
        cdef FixedPartialHeadConfig config = FixedPartialHeadConfig.__new__(FixedPartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_dynamic_partial_heads(self) -> DynamicPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available labels
        that is determined dynamically. Only those labels for which the square of the predictive quality exceeds a
        certain threshold are included in a rule head.

        :return: A `DynamicPartialHeadConfig` that allows further configuration of the rule heads
        """
        cdef IDynamicPartialHeadConfig* config_ptr = &self.config_ptr.get().useDynamicPartialHeads()
        cdef DynamicPartialHeadConfig config = DynamicPartialHeadConfig.__new__(DynamicPartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        self.config_ptr.get().useSingleLabelHeads()

    def use_automatic_statistics(self):
        """
        Configures the rule learner to automatically decide whether a dense or sparse representation of gradients and
        Hessians should be used.
        """
        self.config_ptr.get().useAutomaticStatistics()

    def use_dense_statistics(self):
        """
        Configures the rule learner to use a dense representation of gradients and Hessians.
        """
        self.config_ptr.get().useDenseStatistics()

    def use_sparse_statistics(self):
        """
        Configures the rule learner to use a sparse representation of gradients and Hessians, if possible.
        """
        self.config_ptr.get().useSparseStatistics()

    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """
        self.config_ptr.get().useNoL1Regularization()

    def use_l1_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L1 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IManualRegularizationConfig* config_ptr = &self.config_ptr.get().useL1Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
        self.config_ptr.get().useNoL2Regularization()

    def use_l2_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L2 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IManualRegularizationConfig* config_ptr = &self.config_ptr.get().useL2Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied example-wise.
        """
        self.config_ptr.get().useExampleWiseLogisticLoss()

    def use_example_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied example-wise.
        """
        self.config_ptr.get().useExampleWiseSquaredErrorLoss()

    def use_example_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied example-wise.
        """
        self.config_ptr.get().useExampleWiseSquaredHingeLoss()

    def use_label_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.
        """
        self.config_ptr.get().useLabelWiseLogisticLoss()

    def use_label_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied label-wise.
        """
        self.config_ptr.get().useLabelWiseSquaredErrorLoss()

    def use_label_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied label-wise.
        """
        self.config_ptr.get().useLabelWiseSquaredHingeLoss()

    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
        self.config_ptr.get().useNoLabelBinning()

    def use_automatic_label_binning(self):
        """
        Configures the rule learner to automatically decide whether a method for the assignment of labels to bins should
        be used or not.
        """
        self.config_ptr.get().useAutomaticLabelBinning()

    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
        cdef IEqualWidthLabelBinningConfig* config_ptr = &self.config_ptr.get().useEqualWidthLabelBinning()
        cdef EqualWidthLabelBinningConfig config = EqualWidthLabelBinningConfig.__new__(EqualWidthLabelBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_binary_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model
        and transforming them into binary values according to a certain threshold that is applied to each label
        individually.
        """
        self.config_ptr.get().useLabelWiseBinaryPredictor()

    def use_example_wise_binary_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by an existing rule-based model and comparing the
        aggregated score vector to the known label vectors according to a certain distance measure. The label vector
        that is closest to the aggregated score vector is finally predicted.
        """
        self.config_ptr.get().useExampleWiseBinaryPredictor()

    def use_gfm_binary_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by the individual rules of a existing rule-based model and
        transforming them into binary values according to the general F-measure maximizer (GFM).
        """
        self.config_ptr.get().useGfmBinaryPredictor()

    def use_automatic_binary_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting whether individual labels are
        relevant or irrelevant.
        """
        self.config_ptr.get().useLabelWiseBinaryPredictor()

    def use_label_wise_score_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting regression scores by summing up the scores that
        are provided by the individual rules of an existing rule-based model for each label individually.
        """
        self.config_ptr.get().useLabelWiseScorePredictor()

    def use_label_wise_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and transforming the aggregated scores
        into probabilities according to a certain transformation function that is applied to each label individually.
        """
        self.config_ptr.get().useLabelWiseProbabilityPredictor()

    def use_marginalized_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
        to the known label vectors according to a certain distance measure. The probability for an individual label
        calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
        is specified to be relevant, divided by the total sum of all distances.
        """
        self.config_ptr.get().useMarginalizedProbabilityPredictor()

    def use_automatic_probability_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting probability estimates.
        """
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
