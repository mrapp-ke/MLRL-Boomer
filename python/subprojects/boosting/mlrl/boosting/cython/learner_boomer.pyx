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


cdef class BoomerConfig(BoostingRuleLearnerConfig):
    """
    Allows to configure the BOOMER algorithm.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createBoomerConfig()

    cdef IBoostingRuleLearnerConfig* get_boosting_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    def use_sequential_rule_model_assemblage(self):
        """
        Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
        with a default rule, that are added to a rule-based model.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSequentialRuleModelAssemblage()

    def use_default_rule(self):
        """
        Configures the rule learner to induce a default rule.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useDefaultRule()

    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a greedy top-down search for the induction of individual rules.

        :return: A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IGreedyTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useGreedyTopDownRuleInduction()
        cdef GreedyTopDownRuleInductionConfig config = GreedyTopDownRuleInductionConfig.__new__(GreedyTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a top-down beam search for the induction of individual rules.

        :return: A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IBeamSearchTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useBeamSearchTopDownRuleInduction()
        cdef BeamSearchTopDownRuleInductionConfig config = BeamSearchTopDownRuleInductionConfig.__new__(BeamSearchTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_post_processor(self):
        """
        Configures the rule learner to not use any post-processor.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoPostProcessor()

    def use_no_feature_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoFeatureBinning()

    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains values from equally sized value ranges.

        :return: An `EqualWidthFeatureBinningConfig` that allows further configuration of the method for the assignment
                 of numerical feature values to bins
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IEqualWidthFeatureBinningConfig* config_ptr = &rule_learner_config_ptr.useEqualWidthFeatureBinning()
        cdef EqualWidthFeatureBinningConfig config = EqualWidthFeatureBinningConfig.__new__(EqualWidthFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_equal_frequency_feature_binning(self) -> EqualFrequencyFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains approximately the same number of values.

        :return: An `EqualFrequencyFeatureBinningConfig` that allows further configuration of the method for the
                 assignment of numerical feature values to bins
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IEqualFrequencyFeatureBinningConfig* config_ptr = &rule_learner_config_ptr.useEqualFrequencyFeatureBinning()
        cdef EqualFrequencyFeatureBinningConfig config = EqualFrequencyFeatureBinningConfig.__new__(EqualFrequencyFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_sampling(self):
        """
        Configures the rule learner to not sample from the available labels whenever a new rule should be learned.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available labels with replacement whenever a new rule should be
        learned.

        :return: A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the method for sampling
                 labels
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        """
        Configures the rule learner to not sample from the available training examples whenever a new rule should be
        learned.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoInstanceSampling()
    
    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples with replacement whenever a new rule
        should be learned.

        :return: An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method for sampling
                 instances
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IInstanceSamplingWithReplacementConfig* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithReplacement()
        cdef InstanceSamplingWithReplacementConfig config = InstanceSamplingWithReplacementConfig.__new__(InstanceSamplingWithReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples without replacement whenever a new
        rule should be learned.

        :return: An `InstanceSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling instances
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IInstanceSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithoutReplacement()
        cdef InstanceSamplingWithoutReplacementConfig config = InstanceSamplingWithoutReplacementConfig.__new__(InstanceSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_instance_sampling(self) -> LabelWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, such that for
        each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should be
        learned.

        :return: A `LabelWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseStratifiedInstanceSamplingConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedInstanceSampling()
        cdef LabelWiseStratifiedInstanceSamplingConfig config = LabelWiseStratifiedInstanceSamplingConfig.__new__(LabelWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, where distinct
        label vectors are treated as individual classes, whenever a new rule should be learned.

        :return: An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        """
        Configures the rule learner to not sample from the available features whenever a rule should be refined.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available features with replacement whenever a rule should be
        refined.

        :return: A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling features
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFeatureSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        """
        Configures the rule learner to not partition the available training examples into a training set and a holdout
        set.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoPartitionSampling()
    
    def use_automatic_partition_sampling(self):
        """
        Configures the rule learner to automatically decide whether a holdout set should be used or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticPartitionSampling()

    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        by randomly splitting the training examples into two mutually exclusive sets.

        :return: A `RandomBiPartitionSamplingConfig` that allows further configuration of the method for partitioning
                 the available training examples into a training set and a holdout set
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IRandomBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useRandomBiPartitionSampling()
        cdef RandomBiPartitionSamplingConfig config = RandomBiPartitionSamplingConfig.__new__(RandomBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_bi_partition_sampling(self) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.

        :return: A `LabelWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseStratifiedBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedBiPartitionSampling()
        cdef LabelWiseStratifiedBiPartitionSamplingConfig config = LabelWiseStratifiedBiPartitionSamplingConfig.__new__(LabelWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, where distinct label vectors are treated as individual classes

        :return: An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_rule_pruning(self):
        """
        Configures the rule learner to not prune individual rules.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoRulePruning()

    def use_irep_rule_pruning(self):
        """
        Configures the rule learner to prune individual rules by following the principles of "incremental reduced error
        pruning" (IREP).
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useIrepRulePruning()

    def use_no_parallel_rule_refinement(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelRuleRefinement()
        
    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel refinement of rules.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelRuleRefinement()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_statistic_update(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel update of statistics.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelStatisticUpdate()

    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel update of statistics.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelStatisticUpdate()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_prediction(self):
        """
        Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelPrediction()

    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading to predict for several query examples in parallel.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelPrediction()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_size_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules does
        not exceed a certain maximum.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSizeStoppingCriterion()

    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that the number of induced rules does not
        exceed a certain maximum.

        :return: A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISizeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useSizeStoppingCriterion()
        cdef SizeStoppingCriterionConfig config = SizeStoppingCriterionConfig.__new__(SizeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_time_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is not
        exceeded.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoTimeStoppingCriterion()

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not exceeded.

        :return: A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ITimeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useTimeStoppingCriterion()
        cdef TimeStoppingCriterionConfig config = TimeStoppingCriterionConfig.__new__(TimeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_global_pre_pruning(self) -> PrePruningConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the quality
        of a model's predictions for the examples in the training or holdout set do not improve according to a certain
        measure.

        :return: A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPrePruningConfig* config_ptr = &rule_learner_config_ptr.useGlobalPrePruning()
        cdef PrePruningConfig config = PrePruningConfig.__new__(PrePruningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_global_pruning(self):
        """
        Configures the rule learner to not use global pruning.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoGlobalPruning()

    def use_global_post_pruning(self) -> PostPruningConfig:
        """
        Configures the rule learner to use a stopping criterion that keeps track of the number of rules in a model that
        perform best with respect to the examples in the training or holdout set according to a certain measure.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPostPruningConfig* config_ptr = &rule_learner_config_ptr.useGlobalPostPruning()
        cdef PostPruningConfig config = PostPruningConfig.__new__(PostPruningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_sequential_post_optimization(self):
        """
        Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
        relearning it in the context of the other rules.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSequentialPostOptimization()

    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        """
        Configures the rule learner to use a post-optimization method that optimizes each rule in a model by relearning
        it in the context of the other rules.

        :return: A `SequentialPostOptimizationConfig` that allows further configuration of the post-optimization method
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISequentialPostOptimizationConfig* config_ptr = &rule_learner_config_ptr.useSequentialPostOptimization()
        cdef SequentialPostOptimizationConfig config = SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_default_rule(self):
        """
        Configures the rule learner to not induce a default rule.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoDefaultRule()

    def use_automatic_default_rule(self):
        """
        Configures the rule learner to automatically decide whether a default rule should be induced or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticDefaultRule()

    def use_automatic_feature_binning(self):
        """
        Configures the rule learning to automatically decide whether a method for the assignment of numerical feature
        values to bins should be used or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticFeatureBinning()

    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IConstantShrinkageConfig* config_ptr = &rule_learner_config_ptr.useConstantShrinkagePostProcessor()
        cdef ConstantShrinkageConfig config = ConstantShrinkageConfig.__new__(ConstantShrinkageConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_parallel_rule_refinement(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        refinement of rules or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticParallelRuleRefinement()

    def use_automatic_parallel_statistic_update(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        update of statistics or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticParallelStatisticUpdate()

    def use_automatic_heads(self):
        """
        Configures the rule learner to automatically decide for the type of rule heads to be used.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticHeads()

    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available labels.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useCompleteHeads()

    def use_fixed_partial_heads(self) -> FixedPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a predefined number of labels.

        :return: A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFixedPartialHeadConfig* config_ptr = &rule_learner_config_ptr.useFixedPartialHeads()
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
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IDynamicPartialHeadConfig* config_ptr = &rule_learner_config_ptr.useDynamicPartialHeads()
        cdef DynamicPartialHeadConfig config = DynamicPartialHeadConfig.__new__(DynamicPartialHeadConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSingleLabelHeads()

    def use_automatic_statistics(self):
        """
        Configures the rule learner to automatically decide whether a dense or sparse representation of gradients and
        Hessians should be used.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticStatistics()

    def use_dense_statistics(self):
        """
        Configures the rule learner to use a dense representation of gradients and Hessians.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useDenseStatistics()

    def use_sparse_statistics(self):
        """
        Configures the rule learner to use a sparse representation of gradients and Hessians, if possible.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSparseStatistics()

    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoL1Regularization()

    def use_l1_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L1 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualRegularizationConfig* config_ptr = &rule_learner_config_ptr.useL1Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoL2Regularization()

    def use_l2_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L2 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualRegularizationConfig* config_ptr = &rule_learner_config_ptr.useL2Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied example-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseLogisticLoss()

    def use_example_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied example-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseSquaredErrorLoss()

    def use_example_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied example-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseSquaredHingeLoss()

    def use_label_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseLogisticLoss()

    def use_label_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied label-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseSquaredErrorLoss()

    def use_label_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied label-wise.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseSquaredHingeLoss()

    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelBinning()

    def use_automatic_label_binning(self):
        """
        Configures the rule learner to automatically decide whether a method for the assignment of labels to bins should
        be used or not.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticLabelBinning()

    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IEqualWidthLabelBinningConfig* config_ptr = &rule_learner_config_ptr.useEqualWidthLabelBinning()
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
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseBinaryPredictor()

    def use_example_wise_binary_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by an existing rule-based model and comparing the
        aggregated score vector to the known label vectors according to a certain distance measure. The label vector
        that is closest to the aggregated score vector is finally predicted.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseBinaryPredictor()

    def use_gfm_binary_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by the individual rules of a existing rule-based model and
        transforming them into binary values according to the general F-measure maximizer (GFM).
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useGfmBinaryPredictor()

    def use_automatic_binary_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting whether individual labels are
        relevant or irrelevant.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseBinaryPredictor()

    def use_marginalized_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
        to the known label vectors according to a certain distance measure. The probability for an individual label
        calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
        is specified to be relevant, divided by the total sum of all distances.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useMarginalizedProbabilityPredictor()

    def use_automatic_probability_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting probability estimates.
        """
        cdef IBoomerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticProbabilityPredictor()


cdef class Boomer(RuleLearner):
    """
    The BOOMER rule learning algorithm.
    """

    def __cinit__(self, BoomerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createBoomer(move(config.rule_learner_config_ptr), ddot, dspmv, dsysv)

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
