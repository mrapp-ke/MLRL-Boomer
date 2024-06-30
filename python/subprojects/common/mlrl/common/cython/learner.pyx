"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.common.cython.feature_binning import EqualFrequencyFeatureBinningConfig, EqualWidthFeatureBinningConfig
from mlrl.common.cython.feature_sampling import FeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling import ExampleWiseStratifiedInstanceSamplingConfig, \
    InstanceSamplingWithoutReplacementConfig, InstanceSamplingWithReplacementConfig, \
    OutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.multi_threading import ManualMultiThreadingConfig
from mlrl.common.cython.output_sampling import OutputSamplingWithoutReplacementConfig
from mlrl.common.cython.partition_sampling import ExampleWiseStratifiedBiPartitionSamplingConfig, \
    OutputWiseStratifiedBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization import SequentialPostOptimizationConfig
from mlrl.common.cython.rule_induction import BeamSearchTopDownRuleInductionConfig, GreedyTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion import PostPruningConfig, PrePruningConfig, SizeStoppingCriterionConfig, \
    TimeStoppingCriterionConfig


cdef class TrainingResult:
    """
    Provides access to the results of fitting a rule learner to training data. It incorporates the model that has been
    trained, as well as additional information that is necessary for obtaining predictions for unseen data.
    """

    def __cinit__(self, uint32 num_outputs, RuleModel rule_model not None, OutputSpaceInfo output_space_info not None,
                  MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                  JointProbabilityCalibrationModel joint_probability_calibration_model not None):
        """
        :param num_outputs:                             The number of outputs for which a model has been trained
        :param rule_model:                              The `RuleModel` that has been trained
        :param output_space_info:                       The `OutputSpaceInfo` that may be used as a basis for making
                                                        predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        """
        self.num_outputs = num_outputs
        self.rule_model = rule_model
        self.output_space_info = output_space_info
        self.marginal_probability_calibration_model = marginal_probability_calibration_model
        self.joint_probability_calibration_model = joint_probability_calibration_model


cdef class RuleLearnerConfig:
    pass


class SequentialRuleModelAssemblageMixin(ABC):
    """
    Allows to configure a rule learner to use an algorithm that sequentially induces several rules.
    """
    
    @abstractmethod
    def use_sequential_rule_model_assemblage(self):
        """
        Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
        with a default rule, that are added to a rule-based model.
        """
        pass


class DefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to induce a default rule.
    """

    @abstractmethod
    def use_default_rule(self):
        """
        Configures the rule learner to induce a default rule.
        """
        pass


class GreedyTopDownRuleInductionMixin(ABC):
    """
    Allows to configure a rule learner to use a greedy top-down search for the induction of individual rules.
    """

    @abstractmethod
    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a greedy top-down search for the induction of individual rules.

        :return: A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        pass


class BeamSearchTopDownRuleInductionMixin(ABC):
    """
    Allows to configure a rule learner to use a top-down beam search.
    """

    @abstractmethod
    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a top-down beam search for the induction of individual rules.

        :return: A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        pass


class NoPostProcessorMixin(ABC):
    """
    Allows to configure a rule learner to not use any post processor.
    """

    @abstractmethod
    def use_no_post_processor(self):
        """
        Configures the rule learner to not use any post-processor.
        """
        pass


class NoFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to not use any method for the assignment of numerical features values to bins.
    """

    @abstractmethod
    def use_no_feature_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
        """
        pass


class EqualWidthFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to use equal-width feature binning.
    """

    @abstractmethod
    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains values from equally sized value ranges.

        :return: An `EqualWidthFeatureBinningConfig` that allows further configuration of the method for the assignment
                 of numerical feature values to bins
        """
        pass


class EqualFrequencyFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to use equal-frequency feature binning.
    """

    @abstractmethod
    def use_equal_frequency_feature_binning(self) -> EqualFrequencyFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains approximately the same number of values.

        :return: An `EqualFrequencyFeatureBinningConfig` that allows further configuration of the method for the
                 assignment of numerical feature values to bins
        """
        pass


class NoOutputSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not use output sampling.
    """

    @abstractmethod
    def use_no_output_sampling(self):
        """
        Configures the rule learner to not sample from the available outputs whenever a new rule should be learned.
        """
        pass


class RoundRobinOutputSamplingMixin(ABC):
    """
    Allows to configure a rule learner to sample one output at a time in a round-robin fashion.
    """

    @abstractmethod
    def use_round_robin_output_sampling(self):
        """
        Configures the rule learner to sample a one output at a time in a round-robin fashion whenever a new rule should
        be learned.
        """
        pass


class OutputSamplingWithoutReplacementMixin(ABC):
    """
    Allows to configure a rule learner to use output sampling without replacement.
    """

    @abstractmethod
    def use_output_sampling_without_replacement(self) -> OutputSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available outputs with replacement whenever a new rule should be
        learned.

        :return: An `OutputSamplingWithoutReplacementConfig` that allows further configuration of the sampling method
        """
        pass


class NoInstanceSamplingMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to not use instance sampling.
    """

    @abstractmethod
    def use_no_instance_sampling(self):
        """
        Configures the rule learner to not sample from the available training examples whenever a new rule should be
        learned.
        """
        pass


class InstanceSamplingWithReplacementMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to use instance sampling with
    replacement.
    """

    @abstractmethod
    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples with replacement whenever a new rule
        should be learned.

        :return: An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method for sampling
                 instances
        """
        pass


class InstanceSamplingWithoutReplacementMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to use instance sampling without
    replacement.
    """

    @abstractmethod
    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples without replacement whenever a new
        rule should be learned.

        :return: An `InstanceSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class OutputWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use label-wise stratified instance sampling.
    """

    @abstractmethod
    def use_output_wise_stratified_instance_sampling(self) -> OutputWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, such that for
        each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should be
        learned.

        :return: An `OutputWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class ExampleWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use example-wise stratified instance sampling.
    """

    @abstractmethod
    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, where distinct
        label vectors are treated as individual classes, whenever a new rule should be learned.

        :return: An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class NoFeatureSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not use feature sampling.
    """

    @abstractmethod
    def use_no_feature_sampling(self):
        """
        Configures the rule learner to not sample from the available features whenever a rule should be refined.
        """
        pass
        

class FeatureSamplingWithoutReplacementMixin(ABC):
    """
    Allows to configure a rule learner to use feature sampling without replacement.
    """

    @abstractmethod
    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available features with replacement whenever a rule should be
        refined.

        :return: A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling features
        """
        pass


class NoPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not partition the available training examples into a training set and a
    holdout set.
    """

    @abstractmethod
    def use_no_partition_sampling(self):
        """
        Configures the rule learner to not partition the available training examples into a training set and a holdout
        set.
        """
        pass


class RandomBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training example into a training set and a holdout set
    by randomly splitting the training examples into two mutually exclusive sets.
    """

    @abstractmethod
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        by randomly splitting the training examples into two mutually exclusive sets.

        :return: A `RandomBiPartitionSamplingConfig` that allows further configuration of the method for partitioning
                 the available training examples into a training set and a holdout set
        """
        pass


class OutputWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    @abstractmethod
    def use_output_wise_stratified_bi_partition_sampling(self) -> OutputWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.

        :return: An `OutputWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass


class ExampleWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, where distinct label vectors are treated as individual classes.
    """

    @abstractmethod
    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, where distinct label vectors are treated as individual classes

        :return: An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass


class NoRulePruningMixin(ABC):
    """
    Allows to configure a rule learner to not prune individual rules.
    """

    @abstractmethod
    def use_no_rule_pruning(self):
        """
        Configures the rule learner to not prune individual rules.
        """
        pass


class IrepRulePruningMixin(ABC):
    """
    Allows to configure a rule learner to prune individual rules by following the principles of "incremental reduced
    error pruning" (IREP).
    """

    @abstractmethod
    def use_irep_rule_pruning(self):
        """
        Configures the rule learner to prune individual rules by following the principles of "incremental reduced error
        pruning" (IREP).
        """
        pass


class NoParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for the parallel refinement of rules.
    """

    @abstractmethod
    def use_no_parallel_rule_refinement(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
        """
        pass


class ParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading for the parallel refinement of rules.
    """

    @abstractmethod
    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel refinement of rules.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for the parallel update of statistics.
    """

    @abstractmethod
    def use_no_parallel_statistic_update(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel update of statistics.
        """
        pass


class ParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading for the parallel update of statistics.
    """

    @abstractmethod
    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel update of statistics.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoParallelPredictionMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for prediction.
    """

    @abstractmethod
    def use_no_parallel_prediction(self):
        """
        Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
        """
        pass


class ParallelPredictionMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading to predict for several examples in parallel.
    """

    @abstractmethod
    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading to predict for several query examples in parallel.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoSizeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to not use a stopping criterion that ensures that the number of induced rules
    does not exceed a certain maximum.
    """

    @abstractmethod
    def use_no_size_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules does
        not exceed a certain maximum.
        """
        pass


class SizeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that ensures that the number of induced rules does
    not exceed a certain maximum.
    """

    @abstractmethod
    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that the number of induced rules does not
        exceed a certain maximum.

        :return: A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        pass


class NoTimeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to not use a stopping criterion that ensures that a certain time limit is not
    exceeded.
    """

    @abstractmethod
    def use_no_time_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is not
        exceeded.
        """
        pass


class TimeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that ensures that a certain time limit is not
    exceeded.
    """

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not exceeded.

        :return: A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        pass


class PrePruningMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that stops the induction of rules as soon as the
    quality of a model's predictions for the examples in the training or holdout set do not improve according to a
    certain measure.
    """

    @abstractmethod
    def use_global_pre_pruning(self) -> PrePruningConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the quality
        of a model's predictions for the examples in the training or holdout set do not improve according to a certain
        measure.

        :return: A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        pass


class NoGlobalPruningMixin(ABC):
    """
    Allows to configure a rule learner to not use global pruning.
    """

    @abstractmethod
    def use_no_global_pruning(self):
        """
        Configures the rule learner to not use global pruning.
        """
        pass


class PostPruningMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that keeps track of the number of rules in a model
    that perform best with respect to the examples in the training or holdout set according to a certain measure.
    """

    @abstractmethod
    def use_global_post_pruning(self) -> PostPruningConfig:
        """
        Configures the rule learner to use a stopping criterion that keeps track of the number of rules in a model that
        perform best with respect to the examples in the training or holdout set according to a certain measure.
        """
        pass


class NoSequentialPostOptimizationMixin(ABC):
    """
    Allows to configure a rule learner to not use a post-optimization method that optimizes each rule in a model by
    relearning it in the context of the other rules.
    """

    @abstractmethod
    def use_no_sequential_post_optimization(self):
        """
        Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
        relearning it in the context of the other rules.
        """
        pass


class SequentialPostOptimizationMixin(ABC):
    """
    Allows to configure a rule learner to use a post-optimization method that optimizes each rule in a model by
    relearning it in the context of the other rules.
    """

    @abstractmethod
    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        """
        Configures the rule learner to use a post-optimization method that optimizes each rule in a model by relearning
        it in the context of the other rules.

        :return: A `SequentialPostOptimizationConfig` that allows further configuration of the post-optimization method
        """
        pass


class NoMarginalProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to not calibrate marginal probabilities.
    """

    @abstractmethod
    def use_no_marginal_probability_calibration(self):
        """
        Configures the rule learner to not calibrate marginal probabilities.
        """
        pass


class NoJointProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to not calibrate joint probabilities.
    """
     
    @abstractmethod
    def use_no_joint_probability_calibration(self):
        """
        Configures the rule learner to not calibrate joint probabilities.
        """
        pass
