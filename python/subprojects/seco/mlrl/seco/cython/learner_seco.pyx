"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

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
from mlrl.common.cython.rng cimport IRNGConfig, RNGConfig
from mlrl.common.cython.rule_induction cimport BeamSearchTopDownRuleInductionConfig, GreedyTopDownRuleInductionConfig, \
    IBeamSearchTopDownRuleInductionConfig, IGreedyTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion cimport ISizeStoppingCriterionConfig, ITimeStoppingCriterionConfig, \
    SizeStoppingCriterionConfig, TimeStoppingCriterionConfig

from mlrl.seco.cython.heuristic cimport FMeasureConfig, IFMeasureConfig, IMEstimateConfig, MEstimateConfig
from mlrl.seco.cython.lift_function cimport IKlnLiftFunctionConfig, IPeakLiftFunctionConfig, KlnLiftFunctionConfig, \
    PeakLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion cimport CoverageStoppingCriterionConfig, ICoverageStoppingCriterionConfig

from mlrl.common.cython.learner import BeamSearchTopDownRuleInductionMixin, DefaultRuleMixin, \
    EqualFrequencyFeatureBinningMixin, EqualWidthFeatureBinningMixin, FeatureSamplingWithoutReplacementMixin, \
    GreedyTopDownRuleInductionMixin, InstanceSamplingWithoutReplacementMixin, InstanceSamplingWithReplacementMixin, \
    IrepRulePruningMixin, NoFeatureBinningMixin, NoFeatureSamplingMixin, NoInstanceSamplingMixin, \
    NoOutputSamplingMixin, NoParallelPredictionMixin, NoParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    NoPartitionSamplingMixin, NoRulePruningMixin, NoSequentialPostOptimizationMixin, NoSizeStoppingCriterionMixin, \
    NoTimeStoppingCriterionMixin, OutputSamplingWithoutReplacementMixin, ParallelPredictionMixin, \
    ParallelRuleRefinementMixin, ParallelStatisticUpdateMixin, RandomBiPartitionSamplingMixin, RNGMixin, \
    RoundRobinOutputSamplingMixin, SequentialPostOptimizationMixin, SequentialRuleModelAssemblageMixin, \
    SizeStoppingCriterionMixin, TimeStoppingCriterionMixin
from mlrl.common.cython.learner_classification import ExampleWiseStratifiedBiPartitionSamplingMixin, \
    ExampleWiseStratifiedInstanceSamplingMixin, OutputWiseStratifiedBiPartitionSamplingMixin, \
    OutputWiseStratifiedInstanceSamplingMixin

from mlrl.seco.cython.learner import AccuracyHeuristicMixin, AccuracyPruningHeuristicMixin, \
    CoverageStoppingCriterionMixin, FMeasureHeuristicMixin, FMeasurePruningHeuristicMixin, KlnLiftFunctionMixin, \
    LaplaceHeuristicMixin, LaplacePruningHeuristicMixin, MEstimateHeuristicMixin, MEstimatePruningHeuristicMixin, \
    NoCoverageStoppingCriterionMixin, NoLiftFunctionMixin, OutputWiseBinaryPredictionMixin, PartialHeadMixin, \
    PeakLiftFunctionMixin, PrecisionHeuristicMixin, PrecisionPruningHeuristicMixin, RecallHeuristicMixin, \
    RecallPruningHeuristicMixin, SingleOutputHeadMixin, WraHeuristicMixin, WraPruningHeuristicMixin


cdef class SeCoClassifierConfig(RuleLearnerConfig,
                                RNGMixin,
                                NoCoverageStoppingCriterionMixin,
                                CoverageStoppingCriterionMixin,
                                SingleOutputHeadMixin,
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
                                OutputWiseBinaryPredictionMixin,
                                SequentialRuleModelAssemblageMixin,
                                DefaultRuleMixin,
                                GreedyTopDownRuleInductionMixin,
                                BeamSearchTopDownRuleInductionMixin,
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
                                NoSequentialPostOptimizationMixin,
                                SequentialPostOptimizationMixin):
    """
    Allows to configure the multi-label SeCo algorithm.
    """

    def __cinit__(self):
        self.config_ptr = createSeCoClassifierConfig()

    def use_rng(self) -> RNGConfig:
        cdef IRNGConfig* config_ptr = &self.config_ptr.get().useRNG()
        cdef RNGConfig config = RNGConfig.__new__(RNGConfig)
        config.config_ptr = config_ptr
        return config

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

    def use_single_output_heads(self):
        self.config_ptr.get().useSingleOutputHeads()

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

    def use_output_wise_binary_predictor(self):
        self.config_ptr.get().useOutputWiseBinaryPredictor()


cdef class SeCoClassifier(ClassificationRuleLearner):
    """
    The multi-label SeCo algorithm for classification problems.
    """

    def __cinit__(self, SeCoClassifierConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createSeCoClassifier(move(config.config_ptr))

    cdef IClassificationRuleLearner* get_classification_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
