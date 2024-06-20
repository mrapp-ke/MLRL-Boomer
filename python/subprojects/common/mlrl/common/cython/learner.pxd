from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.feature_binning cimport IEqualFrequencyFeatureBinningConfig, IEqualWidthFeatureBinningConfig
from mlrl.common.cython.feature_info cimport IFeatureInfo
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingWithoutReplacementConfig, \
    IInstanceSamplingWithReplacementConfig
from mlrl.common.cython.label_matrix cimport IRowWiseLabelMatrix
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig
from mlrl.common.cython.output_sampling cimport IOutputSamplingWithoutReplacementConfig
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo, OutputSpaceInfo
from mlrl.common.cython.partition_sampling cimport IRandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization cimport ISequentialPostOptimizationConfig
from mlrl.common.cython.prediction cimport IBinaryPredictor, IProbabilityPredictor, IScorePredictor, \
    ISparseBinaryPredictor
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel, JointProbabilityCalibrationModel, MarginalProbabilityCalibrationModel
from mlrl.common.cython.rule_induction cimport IBeamSearchTopDownRuleInductionConfig, IGreedyTopDownRuleInductionConfig
from mlrl.common.cython.rule_model cimport IRuleModel, RuleModel
from mlrl.common.cython.stopping_criterion cimport IPostPruningConfig, IPrePruningConfig, \
    ISizeStoppingCriterionConfig, ITimeStoppingCriterionConfig


cdef extern from "mlrl/common/learner.hpp" nogil:

    cdef cppclass ITrainingResult:

        # Functions:

        uint32 getNumOutputs() const

        unique_ptr[IRuleModel]& getRuleModel()

        unique_ptr[IOutputSpaceInfo]& getOutputSpaceInfo()

        unique_ptr[IMarginalProbabilityCalibrationModel]& getMarginalProbabilityCalibrationModel()

        unique_ptr[IJointProbabilityCalibrationModel]& getJointProbabilityCalibrationModel()


    cdef cppclass ISequentialRuleModelAssemblageMixin"IRuleLearner::ISequentialRuleModelAssemblageMixin":

        # Functions:

        void useSequentialRuleModelAssemblage()


    cdef cppclass IDefaultRuleMixin"IRuleLearner::IDefaultRuleMixin":

        # Functions:

        void useDefaultRule()


    cdef cppclass IGreedyTopDownRuleInductionMixin"IRuleLearner::IGreedyTopDownRuleInductionMixin":

        # Functions:

        IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction()

        
    cdef cppclass IBeamSearchTopDownRuleInductionMixin"IRuleLearner::IBeamSearchTopDownRuleInductionMixin":

        # Functions:

        IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction()


    cdef cppclass INoPostProcessorMixin"IRuleLearner::INoPostProcessorMixin":

        # Functions:

        void useNoPostProcessor()


    cdef cppclass INoFeatureBinningMixin"IRuleLearner::INoFeatureBinningMixin":

        # Functions:

        void useNoFeatureBinning()
        

    cdef cppclass IEqualWidthFeatureBinningMixin"IRuleLearner::IEqualWidthFeatureBinningMixin":

        # Functions:

        IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning()


    cdef cppclass IEqualFrequencyFeatureBinningMixin"IRuleLearner::IEqualFrequencyFeatureBinningMixin":

        # Functions:

        IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning()


    cdef cppclass INoOutputSamplingMixin"IRuleLearner::INoOutputSamplingMixin":

        # Functions:

        void useNoOutputSampling()


    cdef cppclass IRoundRobinOutputSamplingMixin"IRuleLearner::IRoundRobinOutputSamplingMixin":

        # Functions:

        void useRoundRobinOutputSampling()
    
    
    cdef cppclass IOutputSamplingWithoutReplacementMixin"IRuleLearner::IOutputSamplingWithoutReplacementMixin":

        # Functions:

        IOutputSamplingWithoutReplacementConfig& useOutputSamplingWithoutReplacement()


    cdef cppclass INoInstanceSamplingMixin"IRuleLearner::INoInstanceSamplingMixin":

        # Functions:

        void useNoInstanceSampling()


    cdef cppclass IInstanceSamplingWithoutReplacementMixin"IRuleLearner::IInstanceSamplingWithoutReplacementMixin":

        # Functions:

        IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement()


    cdef cppclass IInstanceSamplingWithReplacementMixin"IRuleLearner::IInstanceSamplingWithReplacementMixin":

        # Functions:

        IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement()


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


    cdef cppclass INoRulePruningMixin"IRuleLearner::INoRulePruningMixin":

        # Functions:

        void useNoRulePruning()


    cdef cppclass IIrepRulePruningMixin"IRuleLearner::IIrepRulePruningMixin":

        # Functions:

        void useIrepRulePruning()


    cdef cppclass INoParallelRuleRefinementMixin"IRuleLearner::INoParallelRuleRefinementMixin":

        # Functions:

        void useNoParallelRuleRefinement()


    cdef cppclass IParallelRuleRefinementMixin"IRuleLearner::IParallelRuleRefinementMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelRuleRefinement()

    
    cdef cppclass INoParallelStatisticUpdateMixin"IRuleLearner::INoParallelStatisticUpdateMixin":

        # Functions:

        void useNoParallelStatisticUpdate()


    cdef cppclass IParallelStatisticUpdateMixin"IRuleLearner::IParallelStatisticUpdateMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelStatisticUpdate()


    cdef cppclass INoParallelPredictionMixin"IRuleLearner::INoParallelPredictionMixin":

        # Functions:

        void useNoParallelPrediction()

        
    cdef cppclass IParallelPredictionMixin"IRuleLearner::IParallelPredictionMixin":

        # Functions:

        IManualMultiThreadingConfig& useParallelPrediction()


    cdef cppclass INoSizeStoppingCriterionMixin"IRuleLearner::INoSizeStoppingCriterionMixin":

        # Functions:

        void useNoSizeStoppingCriterion()


    cdef cppclass ISizeStoppingCriterionMixin"IRuleLearner::ISizeStoppingCriterionMixin":

        # Functions:

        ISizeStoppingCriterionConfig& useSizeStoppingCriterion()


    cdef cppclass INoTimeStoppingCriterionMixin"IRuleLearner::INoTimeStoppingCriterionMixin":

        # Functions:

        void useNoTimeStoppingCriterion()


    cdef cppclass ITimeStoppingCriterionMixin"IRuleLearner::ITimeStoppingCriterionMixin":

        # Functions:

        ITimeStoppingCriterionConfig& useTimeStoppingCriterion()


    cdef cppclass IPrePruningMixin"IRuleLearner::IPrePruningMixin":

        # Functions:

        IPrePruningConfig& useGlobalPrePruning()


    cdef cppclass INoGlobalPruningMixin"IRuleLearner::INoGlobalPruningMixin":

        # Functions:

        void useNoGlobalPruning()


    cdef cppclass IPostPruningMixin"IRuleLearner::IPostPruningMixin":

        # Functions:

        IPostPruningConfig& useGlobalPostPruning()


    cdef cppclass INoSequentialPostOptimizationMixin"IRuleLearner::INoSequentialPostOptimizationMixin":

        # Functions:

        void useNoSequentialPostOptimization()


    cdef cppclass ISequentialPostOptimizationMixin"IRuleLearner::ISequentialPostOptimizationMixin":

        # Functions:

        ISequentialPostOptimizationConfig& useSequentialPostOptimization()


    cdef cppclass INoMarginalProbabilityCalibrationMixin"IRuleLearner::INoMarginalProbabilityCalibrationMixin":

        # Functions:

        void useNoMarginalProbabilityCalibration()


    cdef cppclass INoJointProbabilityCalibrationMixin"IRuleLearner::INoJointProbabilityCalibrationMixin":

        # Functions:

        void useNoJointProbabilityCalibration()


    cdef cppclass IRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const IFeatureInfo& featureInfo, const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IBinaryPredictor] createBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        unique_ptr[ISparseBinaryPredictor] createSparseBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo, uint32 numLabels) except +

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IProbabilityPredictor] createProbabilityPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +


cdef class TrainingResult:

    # Attributes:

    cdef readonly uint32 num_outputs

    cdef readonly RuleModel rule_model

    cdef readonly OutputSpaceInfo output_space_info

    cdef readonly MarginalProbabilityCalibrationModel marginal_probability_calibration_model
    
    cdef readonly JointProbabilityCalibrationModel joint_probability_calibration_model


cdef class RuleLearnerConfig:
    
    # Attributes:

    cdef dict __dict__


cdef class RuleLearner:

    # Functions:

    cdef IRuleLearner* get_rule_learner_ptr(self)
