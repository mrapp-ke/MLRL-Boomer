"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
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
    ITimeStoppingCriterionConfig, TimeStoppingCriterionConfig
from mlrl.seco.cython.heuristic cimport IFMeasureConfig, FMeasureConfig, IMEstimateConfig, MEstimateConfig
from mlrl.seco.cython.lift_function cimport IPeakLiftFunctionConfig, PeakLiftFunctionConfig, IKlnLiftFunctionConfig, \
    KlnLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport ICoverageStoppingCriterionConfig, CoverageStoppingCriterionConfig

from libcpp.utility cimport move


cdef class MultiLabelSeCoRuleLearnerConfig(SeCoRuleLearnerConfig):
    """
    Allows to configure the multi-label SeCo algorithm.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createMultiLabelSeCoRuleLearnerConfig()

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    def use_sequential_rule_model_assemblage(self):
        """
        Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
        with a default rule, that are added to a rule-based model.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSequentialRuleModelAssemblage()

    def use_default_rule(self):
        """
        Configures the rule learner to induce a default rule.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useDefaultRule()

    def use_no_post_processor(self):
        """
        Configures the rule learner to not use any post-processor.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoPostProcessor()

    def use_no_feature_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoFeatureBinning()

    def use_no_coverage_stopping_criterion(self):
        """
        Configures the rule learner to not use any stopping criterion that stops the induction of rules as soon as the
        sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoCoverageStoppingCriterion()

    def use_coverage_stopping_criterion(self) -> CoverageStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the sum of
        the weights of the uncovered labels is smaller or equal to a certain threshold.

        :return: A `CoverageStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ICoverageStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useCoverageStoppingCriterion()
        cdef CoverageStoppingCriterionConfig config = CoverageStoppingCriterionConfig.__new__(CoverageStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSingleLabelHeads()

    def use_partial_heads(self):
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available
        labels.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.usePartialHeads()

    def use_no_lift_function(self):
        """
        Configures the rule learner to not use a lift function.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLiftFunction()

    def use_peak_lift_function(self) -> PeakLiftFunctionConfig:
        """
        Configures the rule learner to use a lift function that monotonously increases until a certain number of labels,
        where the maximum lift is reached, and monotonously decreases afterwards.

        :return: A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IPeakLiftFunctionConfig* config_ptr = &rule_learner_config_ptr.usePeakLiftFunction()
        cdef PeakLiftFunctionConfig config = PeakLiftFunctionConfig.__new__(PeakLiftFunctionConfig)
        config.config_ptr = config_ptr
        return config

    def use_kln_lift_function(self) -> KlnLiftFunctionConfig:
        """
        Configures the rule learner to use a lift function that monotonously increases according to the natural
        logarithm of the number of labels for which a rule predicts.

        :return: A `KlnLiftFunctionConfig` that allows further configuration of the lift function
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IKlnLiftFunctionConfig* config_ptr = &rule_learner_config_ptr.useKlnLiftFunction()
        cdef KlnLiftFunctionConfig config = KlnLiftFunctionConfig.__new__(KlnLiftFunctionConfig)
        config.config_ptr = config_ptr
        return config

    def use_accuracy_heuristic(self):
        """
        Configures the rule learner to use the "Accuracy" heuristic for learning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAccuracyHeuristic()

    def use_accuracy_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAccuracyPruningHeuristic()

    def use_f_measure_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for learning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFMeasureConfig* config_ptr = &rule_learner_config_ptr.useFMeasureHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_f_measure_pruning_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for pruning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFMeasureConfig* config_ptr = &rule_learner_config_ptr.useFMeasurePruningHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for learning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IMEstimateConfig* config_ptr = &rule_learner_config_ptr.useMEstimateHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_pruning_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IMEstimateConfig* config_ptr = &rule_learner_config_ptr.useMEstimatePruningHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_laplace_heuristic(self):
        """
        Configures the rule learner to use the "Laplace" heuristic for learning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLaplaceHeuristic()

    def use_laplace_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Laplace" heuristic for pruning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLaplacePruningHeuristic()

    def use_precision_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for learning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.usePrecisionHeuristic()
    
    def use_precision_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for pruning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.usePrecisionPruningHeuristic()

    def use_recall_heuristic(self):
        """
        Configures the rule learner to use the "Recall" heuristic for learning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useRecallHeuristic()

    def use_recall_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Recall" heuristic for pruning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useRecallPruningHeuristic()

    def use_wra_heuristic(self):
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useWraHeuristic()

    def use_wra_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useWraPruningHeuristic()

    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a greedy top-down search for the induction of individual rules.

        :return: A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IBeamSearchTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useBeamSearchTopDownRuleInduction()
        cdef BeamSearchTopDownRuleInductionConfig config = BeamSearchTopDownRuleInductionConfig.__new__(BeamSearchTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_sampling(self):
        """
        Configures the rule learner to not sample from the available labels whenever a new rule should be learned.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available labels with replacement whenever a new rule should be
        learned.

        :return: A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the method for sampling
                 labels
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        """
        Configures the rule learner to not sample from the available training examples whenever a new rule should be
        learned.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoInstanceSampling()

    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples with replacement whenever a new rule
        should be learned.

        :return: An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method for sampling
                 instances
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        """
        Configures the rule learner to not sample from the available features whenever a rule should be refined.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available features with replacement whenever a rule should be
        refined.

        :return: A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling features
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFeatureSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        """
        Configures the rule learner to not partition the available training examples into a training set and a holdout
        set.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoPartitionSampling()
    
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        by randomly splitting the training examples into two mutually exclusive sets.

        :return: A `RandomBiPartitionSamplingConfig` that allows further configuration of the method for partitioning
                 the available training examples into a training set and a holdout set
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_rule_pruning(self):
        """
        Configures the rule learner to not prune individual rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoRulePruning()

    def use_irep_rule_pruning(self):
        """
        Configures the rule learner to prune individual rules by following the principles of "incremental reduced error
        pruning" (IREP).
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useIrepRulePruning()

    def use_no_parallel_rule_refinement(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelRuleRefinement()

    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel refinement of rules.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelRuleRefinement()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_statistic_update(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel update of statistics.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelStatisticUpdate()

    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel update of statistics.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelStatisticUpdate()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_prediction(self):
        """
        Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelPrediction()

    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading to predict for several query examples in parallel.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelPrediction()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_size_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules does
        not exceed a certain maximum.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSizeStoppingCriterion()

    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that the number of induced rules does not
        exceed a certain maximum.

        :return: A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISizeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useSizeStoppingCriterion()
        cdef SizeStoppingCriterionConfig config = SizeStoppingCriterionConfig.__new__(SizeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_time_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is not
        exceeded.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoTimeStoppingCriterion()

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not exceeded.

        :return: A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ITimeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useTimeStoppingCriterion()
        cdef TimeStoppingCriterionConfig config = TimeStoppingCriterionConfig.__new__(TimeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_global_pruning(self):
        """
        Configures the rule learner to not use global pruning.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoGlobalPruning()

    def use_no_sequential_post_optimization(self):
        """
        Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
        relearning it in the context of the other rules.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSequentialPostOptimization()

    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        """
        Configures the rule learner to use a post-optimization method that optimizes each rule in a model by relearning
        it in the context of the other rules.

        :return: A `SequentialPostOptimizationConfig` that allows further configuration of the post-optimization method
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISequentialPostOptimizationConfig* config_ptr = &rule_learner_config_ptr.useSequentialPostOptimization()
        cdef SequentialPostOptimizationConfig config = SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config


cdef class MultiLabelSeCoRuleLearner(RuleLearner):
    """
    The multi-label SeCo algorithm.
    """

    def __cinit__(self, MultiLabelSeCoRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createMultiLabelSeCoRuleLearner(move(config.rule_learner_config_ptr))

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
