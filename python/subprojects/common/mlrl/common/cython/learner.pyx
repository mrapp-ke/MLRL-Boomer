"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._arrays cimport array_uint32, c_matrix_uint8, c_matrix_float64
from mlrl.common.cython.feature_binning cimport EqualWidthFeatureBinningConfig, EqualFrequencyFeatureBinningConfig
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.feature_sampling cimport FeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport ExampleWiseStratifiedInstanceSamplingConfig, \
    LabelWiseStratifiedInstanceSamplingConfig, InstanceSamplingWithReplacementConfig, \
    InstanceSamplingWithoutReplacementConfig
from mlrl.common.cython.label_matrix cimport RowWiseLabelMatrix
from mlrl.common.cython.label_sampling cimport LabelSamplingWithoutReplacementConfig
from mlrl.common.cython.label_space_info cimport create_label_space_info
from mlrl.common.cython.nominal_feature_mask cimport NominalFeatureMask
from mlrl.common.cython.partition_sampling cimport ExampleWiseStratifiedBiPartitionSamplingConfig, \
    LabelWiseStratifiedBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.pruning cimport IrepConfig
from mlrl.common.cython.rule_induction cimport TopDownRuleInductionConfig
from mlrl.common.cython.rule_model cimport create_rule_model
from mlrl.common.cython.stopping_criterion cimport SizeStoppingCriterionConfig, TimeStoppingCriterionConfig, \
    MeasureStoppingCriterionConfig

from libcpp.utility cimport move

from cython.operator cimport dereference

from scipy.sparse import csr_matrix
import numpy as np


cdef class TrainingResult:
    """
    A wrapper for the pure virtual C++ class `ITrainingResult`.
    """

    def __cinit__(self, uint32 num_labels, RuleModel rule_model not None, LabelSpaceInfo label_space_info not None):
        """
        :param num_labels:          The number of labels for which a model has been trained
        :param rule_model:          The `RuleModel` that has been trained
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for making predictions
        """
        self.num_labels = num_labels
        self.rule_model = rule_model
        self.label_space_info = label_space_info


cdef class RuleLearnerConfig:
    """
    A wrapper for the pure virtual C++ class `IRuleLearner::IConfig`.
    """

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        pass

    def use_top_down_rule_induction(self) -> TopDownRuleInductionConfig:
        """
        Configures the algorithm to use a top-down greedy search for the induction of individual rules.

        :return: A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for the induction of
                 individual rules
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef TopDownRuleInductionConfigImpl* config_ptr = &rule_learner_config_ptr.useTopDownRuleInduction()
        cdef TopDownRuleInductionConfig config = TopDownRuleInductionConfig.__new__(TopDownRuleInductionConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoFeatureBinning()

    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains values from equally sized value ranges.

        :return: An `EqualWidthFeatureBinningConfig` that allows further configuration of the method for the assignment
                 of numerical feature values to bins
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef EqualWidthFeatureBinningConfigImpl* config_ptr = &rule_learner_config_ptr.useEqualWidthFeatureBinning()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef EqualFrequencyFeatureBinningConfigImpl* config_ptr = &rule_learner_config_ptr.useEqualFrequencyFeatureBinning()
        cdef EqualFrequencyFeatureBinningConfig config = EqualFrequencyFeatureBinningConfig.__new__(EqualFrequencyFeatureBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_label_sampling(self):
        """
        Configures the rule learner to not sample from the available labels whenever a new rule should be learned.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoLabelSampling()

    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available labels with replacement whenever a new rule should be
        learned.

        :return: A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the method for sampling
                 labels
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef LabelSamplingWithoutReplacementConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelSamplingWithoutReplacement()
        cdef LabelSamplingWithoutReplacementConfig config = LabelSamplingWithoutReplacementConfig.__new__(LabelSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_instance_sampling(self):
        """
        Configures the rule learner to not sample from the available training examples whenever a new rule should be
        learned.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoInstanceSampling()

    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples with replacement whenever a new rule
        should be learned.

        :return: An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method for sampling
                 instances
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef InstanceSamplingWithReplacementConfigImpl* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithReplacement()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef InstanceSamplingWithoutReplacementConfigImpl* config_ptr = &rule_learner_config_ptr.useInstanceSamplingWithoutReplacement()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef LabelWiseStratifiedInstanceSamplingConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedInstanceSampling()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef ExampleWiseStratifiedInstanceSamplingConfigImpl* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedInstanceSampling()
        cdef ExampleWiseStratifiedInstanceSamplingConfig config = ExampleWiseStratifiedInstanceSamplingConfig.__new__(ExampleWiseStratifiedInstanceSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_feature_sampling(self):
        """
        Configures the rule learner to not sample from the available features whenever a rule should be refined.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoFeatureSampling()

    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available features with replacement whenever a rule should be
        refined.

        :return: A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling features
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef FeatureSamplingWithoutReplacementConfigImpl* config_ptr = &rule_learner_config_ptr.useFeatureSamplingWithoutReplacement()
        cdef FeatureSamplingWithoutReplacementConfig config = FeatureSamplingWithoutReplacementConfig.__new__(FeatureSamplingWithoutReplacementConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_partition_sampling(self):
        """
        Configures the rule learner to not partition the available training examples into a training set and a holdout
        set.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoPartitionSampling()

    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        by randomly splitting the training examples into two mutually exclusive sets.

        :return: A `RandomBiPartitionSamplingConfig` that allows further configuration of the method for partitioning
                 the available training examples into a training set and a holdout set
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef RandomBiPartitionSamplingConfigImpl* config_ptr = &rule_learner_config_ptr.useRandomBiPartitionSampling()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef LabelWiseStratifiedBiPartitionSamplingConfigImpl* config_ptr = &rule_learner_config_ptr.useLabelWiseStratifiedBiPartitionSampling()
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
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfigImpl* config_ptr = &rule_learner_config_ptr.useExampleWiseStratifiedBiPartitionSampling()
        cdef ExampleWiseStratifiedBiPartitionSamplingConfig config = ExampleWiseStratifiedBiPartitionSamplingConfig.__new__(ExampleWiseStratifiedBiPartitionSamplingConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_pruning(self):
        """
        Configures the rule learner to not prune classification rules.
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        rule_learner_config_ptr.useNoPruning()

    def use_irep_pruning(self) -> IrepConfig:
        """
        Configures the rule learner to prune classification rules by following the ideas of "incremental reduced error
        pruning" (IREP).

        :return: An `IrepConfig` that allows further configuration of the method for pruning classification rules
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef IrepConfigImpl* config_ptr = &rule_learner_config_ptr.useIrepPruning()
        cdef IrepConfig config = IrepConfig.__new__(IrepConfig)
        config.config_ptr = config_ptr
        return config

    def add_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        """
        Adds a stopping criterion to the rule learner that ensures that the number of induced rules does not exceed a
        certain minimum.

        :return: A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef SizeStoppingCriterionConfigImpl* config_ptr = &rule_learner_config_ptr.addSizeStoppingCriterion()
        cdef SizeStoppingCriterionConfig config = SizeStoppingCriterionConfig.__new__(SizeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def add_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        """
        Adds a stopping criterion to the rule learner that ensures that a certain time limit is not exceeded.

        :return: A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef TimeStoppingCriterionConfigImpl* config_ptr = &rule_learner_config_ptr.addTimeStoppingCriterion()
        cdef TimeStoppingCriterionConfig config = TimeStoppingCriterionConfig.__new__(TimeStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config

    def add_measure_stopping_criterion(self) -> MeasureStoppingCriterionConfig:
        """
        Adds a stopping criterion to the rule learner that ensures that a stops the induction of rules as soon as the
        quality of a model's predictions for the examples in a holdout set do not improve according to a certain
        measure.

        :return: A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        cdef IRuleLearnerConfig* rule_learner_config_ptr = self.get_rule_learner_config_ptr()
        cdef MeasureStoppingCriterionConfigImpl* config_ptr = &rule_learner_config_ptr.addMeasureStoppingCriterion()
        cdef MeasureStoppingCriterionConfig config = MeasureStoppingCriterionConfig.__new__(MeasureStoppingCriterionConfig)
        config.config_ptr = config_ptr
        return config


cdef class RuleLearner:
    """
    A wrapper for the pure virtual C++ class `IRuleLearner`.
    """

    cdef IRuleLearner* get_rule_learner_ptr(self):
        pass

    def fit(self, NominalFeatureMask nominal_feature_mask not None, ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseLabelMatrix label_matrix not None, uint32 random_state) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth labels.

        :param nominal_feature_mask:    A `NominalFeatureMask` that allows to check whether individual features are
                                        nominal or not
        :param feature_matrix:          A `ColumnWiseFeatureMatrix` that provides column-wise access to the feature
                                        values of the training examples
        :param label_matrix:            A `RowWiseLabelMatrix` that provides row-wise access to the ground truth labels
                                        of the training examples
        :param random_state:            The seed to be used by random number generators
        :return:                        The `TrainingResult` that provides access to the result of fitting the rule
                                        learner to the training data
        """
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_rule_learner_ptr().fit(
            dereference(nominal_feature_mask.get_nominal_feature_mask_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(label_matrix.get_row_wise_label_matrix_ptr()), random_state)
        cdef uint32 num_labels = training_result_ptr.get().getNumLabels()
        cdef unique_ptr[IRuleModel] rule_model_ptr = move(training_result_ptr.get().getRuleModel())
        cdef unique_ptr[ILabelSpaceInfo] label_space_info_ptr = move(training_result_ptr.get().getLabelSpaceInfo())
        cdef RuleModel rule_model = create_rule_model(move(rule_model_ptr))
        cdef LabelSpaceInfo label_space_info = create_label_space_info(move(label_space_info_ptr))
        return TrainingResult.__new__(TrainingResult, num_labels, rule_model, label_space_info)

    def predict_labels(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                       LabelSpaceInfo label_space_info not None, uint32 num_labels) -> np.ndarray:
        """
        Obtains and returns dense predictions for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `numpy.ndarray` of type `uint8`, shape `(num_examples, num_labels)`, that stores
                                    the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = self.get_rule_learner_ptr().predictLabels(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)

    def predict_sparse_labels(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                              LabelSpaceInfo label_space_info not None, uint32 num_labels) -> csr_matrix:
        """
        Obtains and returns sparse predictions for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that
                                    stores the predictions
        """
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = self.get_rule_learner_ptr().predictSparseLabels(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef uint32* row_indices = prediction_matrix_ptr.get().releaseRowIndices()
        cdef uint32* col_indices = prediction_matrix_ptr.get().releaseColIndices()
        cdef uint32 num_non_zero_elements = prediction_matrix_ptr.get().getNumNonZeroElements()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
        indices = np.asarray(array_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
        indptr = np.asarray(array_uint32(row_indices, num_examples + 1))
        return csr_matrix((data, indices, indptr), shape=(num_examples, num_labels))

    def can_predict_probabilities(self) -> bool:
        """
        Returns whether the rule learner is able to predict probability estimates or not.

        :return: True, if the rule learner is able to predict probability estimates, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictProbabilities()

    def predict_probabilities(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                              LabelSpaceInfo label_space_info not None, uint32 num_labels) -> np.ndarray:
        """
        Obtains and returns probability estimates for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that
                                    stores the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.get_rule_learner_ptr().predictProbabilities(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)
