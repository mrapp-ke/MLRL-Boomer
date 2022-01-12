from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython.feature_binning cimport EqualWidthFeatureBinningConfigImpl, \
    EqualFrequencyFeatureBinningConfigImpl
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.feature_sampling cimport FeatureSamplingWithoutReplacementConfigImpl
from mlrl.common.cython.instance_sampling cimport ExampleWiseStratifiedInstanceSamplingConfigImpl, \
    LabelWiseStratifiedInstanceSamplingConfigImpl, InstanceSamplingWithReplacementConfigImpl, \
    InstanceSamplingWithoutReplacementConfigImpl
from mlrl.common.cython.label_matrix cimport IRowWiseLabelMatrix
from mlrl.common.cython.label_sampling cimport LabelSamplingWithoutReplacementConfigImpl
from mlrl.common.cython.label_space_info cimport LabelSpaceInfo, ILabelSpaceInfo
from mlrl.common.cython.nominal_feature_mask cimport INominalFeatureMask
from mlrl.common.cython.partition_sampling cimport ExampleWiseStratifiedBiPartitionSamplingConfigImpl, \
    LabelWiseStratifiedBiPartitionSamplingConfigImpl, RandomBiPartitionSamplingConfigImpl
from mlrl.common.cython.pruning cimport IrepConfigImpl
from mlrl.common.cython.rule_induction cimport TopDownRuleInductionConfigImpl
from mlrl.common.cython.rule_model cimport RuleModel, IRuleModel
from mlrl.common.cython.stopping_criterion cimport SizeStoppingCriterionConfigImpl, TimeStoppingCriterionConfigImpl, \
    MeasureStoppingCriterionConfigImpl

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/output/prediction_matrix_dense.hpp" nogil:

    cdef cppclass DensePredictionMatrix[T]:

        # Functions:

        T* release()


cdef extern from "common/output/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        # Functions:

        uint32 getNumNonZeroElements() const

        uint32* releaseRowIndices()

        uint32* releaseColIndices()


cdef extern from "common/learner.hpp" nogil:

    cdef cppclass ITrainingResult:

        # Functions:

        uint32 getNumLabels() const

        unique_ptr[IRuleModel]& getRuleModel()

        unique_ptr[ILabelSpaceInfo]& getLabelSpaceInfo()


    cdef cppclass IRuleLearnerConfig"IRuleLearner::IConfig":

        # Functions:

        TopDownRuleInductionConfigImpl& useTopDownRuleInduction()

        void useNoFeatureBinning()

        EqualWidthFeatureBinningConfigImpl& useEqualWidthFeatureBinning()

        EqualFrequencyFeatureBinningConfigImpl& useEqualFrequencyFeatureBinning()

        void useNoLabelSampling()

        LabelSamplingWithoutReplacementConfigImpl& useLabelSamplingWithoutReplacement()

        void useNoInstanceSampling()

        ExampleWiseStratifiedInstanceSamplingConfigImpl& useExampleWiseStratifiedInstanceSampling()

        LabelWiseStratifiedInstanceSamplingConfigImpl& useLabelWiseStratifiedInstanceSampling()

        InstanceSamplingWithReplacementConfigImpl& useInstanceSamplingWithReplacement()

        InstanceSamplingWithoutReplacementConfigImpl& useInstanceSamplingWithoutReplacement()

        void useNoFeatureSampling()

        FeatureSamplingWithoutReplacementConfigImpl& useFeatureSamplingWithoutReplacement()

        void useNoPartitionSampling()

        ExampleWiseStratifiedBiPartitionSamplingConfigImpl& useExampleWiseStratifiedBiPartitionSampling()

        LabelWiseStratifiedBiPartitionSamplingConfigImpl& useLabelWiseStratifiedBiPartitionSampling()

        RandomBiPartitionSamplingConfigImpl& useRandomBiPartitionSampling()

        void useNoPruning()

        IrepConfigImpl& useIrepPruning()

        SizeStoppingCriterionConfigImpl& useSizeStoppingCriterion();

        TimeStoppingCriterionConfigImpl& useTimeStoppingCriterion();

        MeasureStoppingCriterionConfigImpl& useMeasureStoppingCriterion();


    cdef cppclass IRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const INominalFeatureMask& nominalFeatureMask,
                                        const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const

        unique_ptr[DensePredictionMatrix[uint8]] predictLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                               const IRuleModel& ruleModel,
                                                               const ILabelSpaceInfo& labelSpaceInfo,
                                                               uint32 numLabels) const

        unique_ptr[BinarySparsePredictionMatrix] predictSparseLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) const

        bool canPredictProbabilities() const

        unique_ptr[DensePredictionMatrix[float64]] predictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const


cdef class TrainingResult:

    # Attributes:

    cdef readonly uint32 num_labels

    cdef readonly RuleModel rule_model

    cdef readonly LabelSpaceInfo label_space_info


cdef class RuleLearnerConfig:

    # Functions:

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self)


cdef class RuleLearner:

    # Functions:

    cdef IRuleLearner* get_rule_learner_ptr(self)
