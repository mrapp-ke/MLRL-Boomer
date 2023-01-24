from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython.feature_binning cimport IEqualWidthFeatureBinningConfig, IEqualFrequencyFeatureBinningConfig
from mlrl.common.cython.feature_info cimport IFeatureInfo
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport IExampleWiseStratifiedInstanceSamplingConfig, \
    ILabelWiseStratifiedInstanceSamplingConfig, IInstanceSamplingWithReplacementConfig, \
    IInstanceSamplingWithoutReplacementConfig
from mlrl.common.cython.label_matrix cimport IRowWiseLabelMatrix
from mlrl.common.cython.label_sampling cimport ILabelSamplingWithoutReplacementConfig
from mlrl.common.cython.label_space_info cimport LabelSpaceInfo, ILabelSpaceInfo
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    ILabelWiseStratifiedBiPartitionSamplingConfig, IRandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization cimport ISequentialPostOptimizationConfig
from mlrl.common.cython.prediction cimport DensePredictionMatrix, BinarySparsePredictionMatrix  # TODO Remove
from mlrl.common.cython.rule_induction cimport IGreedyTopDownRuleInductionConfig, IBeamSearchTopDownRuleInductionConfig
from mlrl.common.cython.rule_model cimport RuleModel, IRuleModel
from mlrl.common.cython.stopping_criterion cimport ISizeStoppingCriterionConfig, ITimeStoppingCriterionConfig, \
    IPrePruningConfig, IPostPruningConfig

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/learner.hpp" nogil:

    cdef cppclass ITrainingResult:

        # Functions:

        uint32 getNumLabels() const

        unique_ptr[IRuleModel]& getRuleModel()

        unique_ptr[ILabelSpaceInfo]& getLabelSpaceInfo()


    cdef cppclass IRuleLearnerConfig"IRuleLearner::IConfig":

        # Functions:

        void useDefaultRule()

        void useSequentialRuleModelAssemblage()

        IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction()

        void useNoFeatureBinning()

        void useNoLabelSampling()

        void useNoInstanceSampling()

        void useNoFeatureSampling()

        void useNoPartitionSampling()

        void useNoRulePruning()

        void useNoPostProcessor()

        void useNoParallelRuleRefinement()

        void useNoParallelStatisticUpdate()

        void useNoParallelPrediction()

        void useNoSizeStoppingCriterion()

        void useNoTimeStoppingCriterion()

        void useNoGlobalPruning()

        void useNoSequentialPostOptimization()


    cdef cppclass IBeamSearchTopDownMixin"IRuleLearner::IBeamSearchTopDownMixin":

        # Functions:

        IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction()


    cdef cppclass IFeatureBinningMixin"IRuleLearner::IFeatureBinningMixin":

        # Functions:

        IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning()

        IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning()


    cdef cppclass ILabelSamplingMixin"IRuleLearner::ILabelSamplingMixin":

        # Functions:

        ILabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement()


    cdef cppclass IInstanceSamplingMixin"IRuleLearner::IInstanceSamplingMixin":

        # Functions:

        IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling()

        ILabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling()

        IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement()

        IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement()


    cdef cppclass IFeatureSamplingMixin"IRuleLearner::IFeatureSamplingMixin":

        # Functions:

        IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement()


    cdef cppclass IPartitionSamplingMixin"IRuleLearner::IPartitionSamplingMixin":

        # Functions:

        IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling()

        ILabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling()

        IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling()


    cdef cppclass IRulePruningMixin"IRuleLearner::IRulePruningMixin":

        # Functions:

        void useIrepRulePruning()


    cdef cppclass IMultiThreadingMixin"IRuleLearner::IMultiThreadingMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelRuleRefinement()

        IManualMultiThreadingConfig& useParallelStatisticUpdate()

        IManualMultiThreadingConfig& useParallelPrediction()


    cdef cppclass ISizeStoppingCriterionMixin"IRuleLearner::ISizeStoppingCriterionMixin":

        # Functions:

        ISizeStoppingCriterionConfig& useSizeStoppingCriterion()


    cdef cppclass ITimeStoppingCriterionMixin"IRuleLearner::ITimeStoppingCriterionMixin":

        # Functions:

        ITimeStoppingCriterionConfig& useTimeStoppingCriterion()


    cdef cppclass IPrePruningMixin"IRuleLearner::IPrePruningMixin":

        # Functions:

        IPrePruningConfig& useGlobalPrePruning()


    cdef cppclass IPostPruningMixin"IRuleLearner::IPostPruningMixin":

        # Functions:

        IPostPruningConfig& useGlobalPostPruning()


    cdef cppclass ISequentialPostOptimizationMixin"IRuleLearner::ISequentialPostOptimizationMixin":

        # Functions:

        ISequentialPostOptimizationConfig& useSequentialPostOptimization()


    cdef cppclass IRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const IFeatureInfo& featureInfo, const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const

        bool canPredictLabels(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[ILabelPredictor] createLabelPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                         const IRuleModel& ruleModel,
                                                         const ILabelSpaceInfo& labelSpaceInfo,
                                                         uint32 numLabels) const

        # TODO Remove
        unique_ptr[DensePredictionMatrix[uint8]] predictLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                               const IRuleModel& ruleModel,
                                                               const ILabelSpaceInfo& labelSpaceInfo,
                                                               uint32 numLabels) const

        unique_ptr[ISparseLabelPredictor] createSparseLabelPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) const

        # TODO Remove
        unique_ptr[BinarySparsePredictionMatrix] predictSparseLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) const

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                         const IRuleModel& ruleModel,
                                                         const ILabelSpaceInfo& labelSpaceInfo,
                                                         uint32 numLabels) const

        # TODO Remove
        unique_ptr[DensePredictionMatrix[float64]] predictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                                                 const IRuleModel& ruleModel,
                                                                 const ILabelSpaceInfo& labelSpaceInfo,
                                                                 uint32 numLabels) const

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IProbabilityPredictor] createProbabilityPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) const

        # TODO Remove
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
