"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
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
from mlrl.common.cython.learner cimport IRuleLearnerConfig
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig, ManualMultiThreadingConfig
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    ExampleWiseStratifiedBiPartitionSamplingConfig, ILabelWiseStratifiedBiPartitionSamplingConfig, \
    LabelWiseStratifiedBiPartitionSamplingConfig, IRandomBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.rule_induction cimport IBeamSearchTopDownRuleInductionConfig, \
    BeamSearchTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion cimport ISizeStoppingCriterionConfig, SizeStoppingCriterionConfig, \
    ITimeStoppingCriterionConfig, TimeStoppingCriterionConfig, IMeasureStoppingCriterionConfig, \
    MeasureStoppingCriterionConfig

from libcpp.utility cimport move


cdef class MultiLabelSeCoRuleLearnerConfig(SeCoRuleLearnerConfig):
    """
    Allows to configure the multi-label SeCo algorithm.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createMultiLabelSeCoRuleLearnerConfig()

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

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

    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains values from equally sized value ranges.

        :return: An `EqualWidthFeatureBinningConfig` that allows further configuration of the method for the assignment
                 of numerical feature values to bins
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
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
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IEqualFrequencyFeatureBinningConfig* config_ptr = &rule_learner_config_ptr.useEqualFrequencyFeatureBinning()
        cdef EqualFrequencyFeatureBinningConfig config = EqualFrequencyFeatureBinningConfig.__new__(EqualFrequencyFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

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

    def use_irep_pruning(self):
        """
        Configures the rule learner to prune classification rules by following the ideas of "incremental reduced error
        pruning" (IREP).
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useIrepPruning()

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

    def use_measure_stopping_criterion(self) -> MeasureStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion stops the induction of rules as soon as the quality of a
        model's predictions for the examples in a holdout set do not improve according to a certain measure.

        :return: A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IMultiLabelSeCoRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IMeasureStoppingCriterionConfig* config_ptr = &rule_learner_config_ptr.useMeasureStoppingCriterion()
        cdef MeasureStoppingCriterionConfig config = MeasureStoppingCriterionConfig.__new__(MeasureStoppingCriterionConfig)
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
