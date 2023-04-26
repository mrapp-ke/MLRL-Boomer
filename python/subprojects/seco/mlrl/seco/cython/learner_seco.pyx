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
from mlrl.seco.cython.learner import NoCoverageStoppingCriterionMixin, CoverageStoppingCriterionMixin, \
    SingleLabelHeadMixin, PartialHeadMixin, NoLiftFunctionMixin, PeakLiftFunctionMixin, KlnLiftFunctionMixin, \
    AccuracyHeuristicMixin, AccuracyPruningHeuristicMixin, FMeasureHeuristicMixin, FMeasurePruningHeuristicMixin, \
    MEstimateHeuristicMixin, MEstimatePruningHeuristicMixin, LaplaceHeuristicMixin, LaplacePruningHeuristicMixin, \
    PrecisionHeuristicMixin, PrecisionPruningHeuristicMixin, RecallHeuristicMixin, RecallPruningHeuristicMixin, \
    WraHeuristicMixin, WraPruningHeuristicMixin, LabelWiseBinaryPredictionMixin



cdef class MultiLabelSeCoRuleLearnerConfig(RuleLearnerConfig,
                                           NoCoverageStoppingCriterionMixin,
                                           CoverageStoppingCriterionMixin,
                                           SingleLabelHeadMixin,
                                           PartialHeadMixin,
                                           NoLiftFunctionMixin,
                                           PeakLiftFunctionMixin,
                                           KlnLiftFunctionMixin,
                                           AccuracyHeuristicMixin,
                                           AccuracyPruningHeuristicMixin,
                                           FMeasureHeuristicMixin,
                                           FMeasurePruningHeuristicMixin,
                                           MEstimateHeuristicMixin,
                                           MEstimatePruningHeuristicMixin,
                                           LaplaceHeuristicMixin,
                                           LaplacePruningHeuristicMixin,
                                           PrecisionHeuristicMixin,
                                           PrecisionPruningHeuristicMixin,
                                           RecallHeuristicMixin,
                                           RecallPruningHeuristicMixin,
                                           WraHeuristicMixin,
                                           WraPruningHeuristicMixin,
                                           LabelWiseBinaryPredictionMixin,
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
        self.config_ptr = createMultiLabelSeCoRuleLearnerConfig()

    def use_sequential_rule_model_assemblage(self):
        self.config_ptr.get().useSequentialRuleModelAssemblage()

    def use_default_rule(self):
        self.config_ptr.get().useDefaultRule()

    def use_no_coverage_stopping_criterion(self):
        self.config_ptr.get().useNoCoverageStoppingCriterion()

    def use_coverage_stopping_criterion(self) -> CoverageStoppingCriterionConfig:
        cdef ICoverageStoppingCriterionConfig* config_ptr = \
            &self.config_ptr.get().useCoverageStoppingCriterion()
        cdef CoverageStoppingCriterionConfig config = \
            CoverageStoppingCriterionConfig.__new__(CoverageStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def use_single_label_heads(self):
        self.config_ptr.get().useSingleLabelHeads()

    def use_partial_heads(self):
        self.config_ptr.get().usePartialHeads()

    def use_no_lift_function(self):
        self.config_ptr.get().useNoLiftFunction()

    def use_peak_lift_function(self) -> PeakLiftFunctionConfig:
        cdef IPeakLiftFunctionConfig* config_ptr = &self.config_ptr.get().usePeakLiftFunction()
        cdef PeakLiftFunctionConfig config = PeakLiftFunctionConfig.__new__(PeakLiftFunctionConfig)
        config.config_ptr = config_ptr
        return config

    def use_kln_lift_function(self) -> KlnLiftFunctionConfig:
        cdef IKlnLiftFunctionConfig* config_ptr = &self.config_ptr.get().useKlnLiftFunction()
        cdef KlnLiftFunctionConfig config = KlnLiftFunctionConfig.__new__(KlnLiftFunctionConfig)
        config.config_ptr = config_ptr
        return config

    def use_accuracy_heuristic(self):
        self.config_ptr.get().useAccuracyHeuristic()

    def use_accuracy_pruning_heuristic(self):
        self.config_ptr.get().useAccuracyPruningHeuristic()

    def use_f_measure_heuristic(self) -> FMeasureConfig:
        cdef IFMeasureConfig* config_ptr = &self.config_ptr.get().useFMeasureHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_f_measure_pruning_heuristic(self) -> FMeasureConfig:
        cdef IFMeasureConfig* config_ptr = &self.config_ptr.get().useFMeasurePruningHeuristic()
        cdef FMeasureConfig config = FMeasureConfig.__new__(FMeasureConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_heuristic(self) -> MEstimateConfig:
        cdef IMEstimateConfig* config_ptr = &self.config_ptr.get().useMEstimateHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_m_estimate_pruning_heuristic(self) -> MEstimateConfig:
        cdef IMEstimateConfig* config_ptr = &self.config_ptr.get().useMEstimatePruningHeuristic()
        cdef MEstimateConfig config = MEstimateConfig.__new__(MEstimateConfig)
        config.config_ptr = config_ptr
        return config

    def use_laplace_heuristic(self):
        self.config_ptr.get().useLaplaceHeuristic()

    def use_laplace_pruning_heuristic(self):
        self.config_ptr.get().useLaplacePruningHeuristic()

    def use_precision_heuristic(self):
        self.config_ptr.get().usePrecisionHeuristic()
    
    def use_precision_pruning_heuristic(self):
        self.config_ptr.get().usePrecisionPruningHeuristic()

    def use_recall_heuristic(self):
        self.config_ptr.get().useRecallHeuristic()

    def use_recall_pruning_heuristic(self):
        self.config_ptr.get().useRecallPruningHeuristic()

    def use_wra_heuristic(self):
        self.config_ptr.get().useWraHeuristic()

    def use_wra_pruning_heuristic(self):
        self.config_ptr.get().useWraPruningHeuristic()

    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        cdef IGreedyTopDownRuleInductionConfig* config_ptr = \
            &self.config_ptr.get().useGreedyTopDownRuleInduction()
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

    def use_no_label_sampling(self):
        self.config_ptr.get().useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        cdef ILabelSamplingWithoutReplacementConfig* config_ptr = \
            &self.config_ptr.get().useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = \
            LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
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

    def use_label_wise_stratified_instance_sampling(self) -> LabelWiseStratifiedInstanceSamplingConfig:
        cdef ILabelWiseStratifiedInstanceSamplingConfig* config_ptr = \
            &self.config_ptr.get().useLabelWiseStratifiedInstanceSampling()
        cdef LabelWiseStratifiedInstanceSamplingConfig config = \
            LabelWiseStratifiedInstanceSamplingConfig.__new__(LabelWiseStratifiedInstanceSamplingConfig)
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

    def use_label_wise_stratified_bi_partition_sampling(self) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        cdef ILabelWiseStratifiedBiPartitionSamplingConfig* config_ptr = \
            &self.config_ptr.get().useLabelWiseStratifiedBiPartitionSampling()
        cdef LabelWiseStratifiedBiPartitionSamplingConfig config = \
            LabelWiseStratifiedBiPartitionSamplingConfig.__new__(LabelWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr = \
            &self.config_ptr.get().useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = \
            ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

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

    def use_no_sequential_post_optimization(self):
        self.config_ptr.get().useNoSequentialPostOptimization()

    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        cdef ISequentialPostOptimizationConfig* config_ptr = &self.config_ptr.get().useSequentialPostOptimization()
        cdef SequentialPostOptimizationConfig config = \
            SequentialPostOptimizationConfig.__new__(SequentialPostOptimizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_label_wise_binary_predictor(self):
        self.config_ptr.get().useLabelWiseBinaryPredictor()


cdef class MultiLabelSeCoRuleLearner(RuleLearner):
    """
    The multi-label SeCo algorithm.
    """

    def __cinit__(self, MultiLabelSeCoRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createMultiLabelSeCoRuleLearner(move(config.config_ptr))

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
