from mlrl.common.cython._types cimport uint32
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
from mlrl.common.cython.prediction cimport IBinaryPredictor, ISparseBinaryPredictor, IScorePredictor, \
    IProbabilityPredictor
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

        void useSequentialRuleModelAssemblage()

        IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction()

        void useNoRulePruning()

        void useNoPostProcessor()

        void useNoParallelStatisticUpdate()

        void useNoParallelPrediction()

        void useNoSizeStoppingCriterion()

        void useNoTimeStoppingCriterion()

        void useNoGlobalPruning()

        void useNoSequentialPostOptimization()


    cdef cppclass IDefaultRuleMixin"IRuleLearner::IDefaultRuleMixin":

        # Functions:

        void useDefaultRule()


    cdef cppclass IBeamSearchTopDownMixin"IRuleLearner::IBeamSearchTopDownMixin":

        # Functions:

        IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction()


    cdef cppclass INoFeatureBinningMixin"IRuleLearner::INoFeatureBinningMixin":

        # Functions:

        void useNoFeatureBinning()
        

    cdef cppclass IEqualWidthFeatureBinningMixin"IRuleLearner::IEqualWidthFeatureBinningMixin":

        # Functions:

        IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning()


    cdef cppclass IEqualFrequencyFeatureBinningMixin"IRuleLearner::IEqualFrequencyFeatureBinningMixin":

        # Functions:

        IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning()


    cdef cppclass INoLabelSamplingMixin"IRuleLearner::INoLabelSamplingMixin":

        # Functions:

        void useNoLabelSampling()


    cdef cppclass ILabelSamplingWithoutReplacementMixin"IRuleLearner::ILabelSamplingWithoutReplacementMixin":

        # Functions:

        ILabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement()


    cdef cppclass INoInstanceSamplingMixin"IRuleLearner::INoInstanceSamplingMixin":

        # Functions:

        void useNoInstanceSampling()


    cdef cppclass IInstanceSamplingWithoutReplacementMixin"IRuleLearner::IInstanceSamplingWithoutReplacementMixin":

        # Functions:

        IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement()


    cdef cppclass IInstanceSamplingWithReplacementMixin"IRuleLearner::IInstanceSamplingWithReplacementMixin":

        # Functions:

        IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement()


    cdef cppclass ILabelWiseStratifiedInstanceSamplingMixin"IRuleLearner::ILabelWiseStratifiedInstanceSamplingMixin":

        # Functions:

        ILabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling()


    cdef cppclass IExampleWiseStratifiedInstanceSamplingMixin"IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin":

        # Functions:

        IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling()


    cdef cppclass INoFeatureSamplingMixin"IRuleLearner::INoFeatureSamplingMixin":

        # Functions:

        void useNoFeatureSampling()


    cdef cppclass IFeatureSamplingWithoutReplacementMixin"IRuleLearner::IFeatureSamplingWithoutReplacementMixin":

        # Functions:

        IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement()


    cdef cppclass INoPartitionSamplingMixin"IRuleLearner::INoPartitionSamplingMixin":

        # Functions:

        void useNoPartitionSampling()


    cdef cppclass IRandomBiPartitionSamplingMixin"IRuleLearner::IRandomBiPartitionSamplingMixin":

        # Functions:

        IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling()


    cdef cppclass ILabelWiseStratifiedBiPartitionSamplingMixin"IRuleLearner::ILabelWiseStratifiedBiPartitionSamplingMixin":

        # Functions:

        ILabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling()


    cdef cppclass IExampleWiseStratifiedBiPartitionSamplingMixin"IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin":

        # Functions:

        IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling()


    cdef cppclass IRulePruningMixin"IRuleLearner::IRulePruningMixin":

        # Functions:

        void useIrepRulePruning()


    cdef cppclass INoParallelRuleRefinementMixin"IRuleLearner::INoParallelRuleRefinementMixin":

        # Functions:

        void useNoParallelRuleRefinement()


    cdef cppclass IParallelRuleRefinementMixin"IRuleLearner::IParallelRuleRefinementMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelRuleRefinement()

    
    cdef cppclass IParallelStatisticUpdateMixin"IRuleLearner::IParallelStatisticUpdateMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelStatisticUpdate()


    cdef cppclass IParallelPredictionMixin"IRuleLearner::IParallelPredictionMixin":

        # Functions:

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

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IBinaryPredictor] createBinaryPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                           const IRuleModel& ruleModel,
                                                           const ILabelSpaceInfo& labelSpaceInfo,
                                                           uint32 numLabels) except +

        unique_ptr[ISparseBinaryPredictor] createSparseBinaryPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                       const IRuleModel& ruleModel,
                                                                       const ILabelSpaceInfo& labelSpaceInfo,
                                                                       uint32 numLabels) except +

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                         const IRuleModel& ruleModel,
                                                         const ILabelSpaceInfo& labelSpaceInfo,
                                                         uint32 numLabels) except +

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IProbabilityPredictor] createProbabilityPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) except +


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
