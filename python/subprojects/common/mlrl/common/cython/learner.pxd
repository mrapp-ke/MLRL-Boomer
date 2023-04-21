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
from mlrl.common.cython.probability_calibration cimport IMarginalProbabilityCalibrationModel, \
    MarginalProbabilityCalibrationModel, IJointProbabilityCalibrationModel, JointProbabilityCalibrationModel
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

        unique_ptr[IMarginalProbabilityCalibrationModel]& getMarginalProbabilityCalibrationModel()

        unique_ptr[IJointProbabilityCalibrationModel]& getJointProbabilityCalibrationModel()


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

        void useNoMarginalProbabilityCalibration()

        void useNoJointProbabilityCalibration()


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

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IBinaryPredictor] createBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        unique_ptr[ISparseBinaryPredictor] createSparseBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) except +

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IProbabilityPredictor] createProbabilityPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +


cdef class TrainingResult:

    # Attributes:

    cdef readonly uint32 num_labels

    cdef readonly RuleModel rule_model

    cdef readonly LabelSpaceInfo label_space_info

    cdef readonly MarginalProbabilityCalibrationModel marginal_probability_calibration_model
    
    cdef readonly JointProbabilityCalibrationModel joint_probability_calibration_model


cdef class RuleLearnerConfig:

    # Functions:

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self)


cdef class RuleLearner:

    # Functions:

    cdef IRuleLearner* get_rule_learner_ptr(self)
