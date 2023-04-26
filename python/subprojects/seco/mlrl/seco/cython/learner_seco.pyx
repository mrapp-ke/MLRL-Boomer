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

from mlrl.common.cython.learner import SequentialRuleModelAssemblageMixin, DefaultRuleMixin, \
    GreedyTopDownRuleInductionMixin, BeamSearchTopDownRuleInductionMixin, EqualWidthFeatureBinningMixin, \
    EqualFrequencyFeatureBinningMixin, NoLabelSamplingMixin, LabelSamplingWithoutReplacementMixin, \
    NoInstanceSamplingMixin, InstanceSamplingWithReplacementMixin, InstanceSamplingWithoutReplacementMixin, \
    LabelWiseStratifiedInstanceSamplingMixin, ExampleWiseStratifiedInstanceSamplingMixin, NoFeatureSamplingMixin, \
    FeatureSamplingWithoutReplacementMixin, NoPartitionSamplingMixin, RandomBiPartitionSamplingMixin, \
    LabelWiseStratifiedBiPartitionSamplingMixin, ExampleWiseStratifiedBiPartitionSamplingMixin, NoRulePruningMixin, \
    IrepRulePruningMixin, NoParallelRuleRefinementMixin, ParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    ParallelStatisticUpdateMixin, NoParallelPredictionMixin, ParallelPredictionMixin, NoSizeStoppingCriterionMixin, \
    SizeStoppingCriterionMixin, NoTimeStoppingCriterionMixin, TimeStoppingCriterionMixin, PrePruningMixin, \
    NoGlobalPruningMixin, PostPruningMixin, NoSequentialPostOptimizationMixin, SequentialPostOptimizationMixin


cdef class MultiLabelSeCoRuleLearnerConfig(RuleLearnerConfig,
                                           SequentialRuleModelAssemblageMixin,
                                           DefaultRuleMixin,
                                           GreedyTopDownRuleInductionMixin,
                                           BeamSearchTopDownRuleInductionMixin,
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
                                           NoSequentialPostOptimizationMixin,
                                           SequentialPostOptimizationMixin):
    """
    Allows to configure the multi-label SeCo algorithm.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createMultiLabelSeCoRuleLearnerConfig()

    def use_sequential_rule_model_assemblage(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSequentialRuleModelAssemblage()

    def use_default_rule(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useDefaultRule()

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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IGreedyTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useGreedyTopDownRuleInduction()
        cdef GreedyTopDownRuleInductionConfig config = GreedyTopDownRuleInductionConfig.__new__(GreedyTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IBeamSearchTopDownRuleInductionConfig* config_ptr = &rule_learner_config_ptr.useBeamSearchTopDownRuleInduction()
        cdef BeamSearchTopDownRuleInductionConfig config = BeamSearchTopDownRuleInductionConfig.__new__(BeamSearchTopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_sampling(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoInstanceSampling()

    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IInstanceSamplingWithReplacementConfig* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithReplacement()
        cdef InstanceSamplingWithReplacementConfig config = InstanceSamplingWithReplacementConfig.__new__(InstanceSamplingWithReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IInstanceSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithoutReplacement()
        cdef InstanceSamplingWithoutReplacementConfig config = InstanceSamplingWithoutReplacementConfig.__new__(InstanceSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_instance_sampling(self) -> LabelWiseStratifiedInstanceSamplingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseStratifiedInstanceSamplingConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedInstanceSampling()
        cdef LabelWiseStratifiedInstanceSamplingConfig config = LabelWiseStratifiedInstanceSamplingConfig.__new__(LabelWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IFeatureSamplingWithoutReplacementConfig* config_ptr = &rule_learner_config_ptr.useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoPartitionSampling()
    
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IRandomBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useRandomBiPartitionSampling()
        cdef RandomBiPartitionSamplingConfig config = RandomBiPartitionSamplingConfig.__new__(RandomBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_stratified_bi_partition_sampling(self) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ILabelWiseStratifiedBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedBiPartitionSampling()
        cdef LabelWiseStratifiedBiPartitionSamplingConfig config = LabelWiseStratifiedBiPartitionSamplingConfig.__new__(LabelWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_rule_pruning(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoRulePruning()

    def use_irep_rule_pruning(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useIrepRulePruning()

    def use_no_parallel_rule_refinement(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelRuleRefinement()

    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelRuleRefinement()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_statistic_update(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelStatisticUpdate()

    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelStatisticUpdate()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_parallel_prediction(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoParallelPrediction()

    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualMultiThreadingConfig* config_ptr = &rule_learner_config_ptr.useParallelPrediction()
        cdef ManualMultiThreadingConfig config = ManualMultiThreadingConfig.__new__(ManualMultiThreadingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_size_stopping_criterion(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSizeStoppingCriterion()

    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISizeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useSizeStoppingCriterion()
        cdef SizeStoppingCriterionConfig config = SizeStoppingCriterionConfig.__new__(SizeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_time_stopping_criterion(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoTimeStoppingCriterion()

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ITimeStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useTimeStoppingCriterion()
        cdef TimeStoppingCriterionConfig config = TimeStoppingCriterionConfig.__new__(TimeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_sequential_post_optimization(self):
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoSequentialPostOptimization()

    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef ISequentialPostOptimizationConfig* config_ptr = &rule_learner_config_ptr.useSequentialPostOptimization()
        cdef SequentialPostOptimizationConfig config = SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_binary_predictor(self):
        """
        Configures the rule learner to use predictor for predicting whether individual labels of given query examples
        are relevant or irrelevant by processing rules of an existing rule-based model in the order they have been
        learned. If a rule covers an example, its prediction is applied to each label individually.
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseBinaryPredictor()


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
