from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.feature_binning cimport IEqualFrequencyFeatureBinningConfig, IEqualWidthFeatureBinningConfig
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingWithoutReplacementConfig, \
    IInstanceSamplingWithReplacementConfig
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig
from mlrl.common.cython.output_sampling cimport IOutputSamplingWithoutReplacementConfig
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo, OutputSpaceInfo
from mlrl.common.cython.partition_sampling cimport IRandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization cimport ISequentialPostOptimizationConfig
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel, JointProbabilityCalibrationModel, MarginalProbabilityCalibrationModel
from mlrl.common.cython.rng cimport IRNGConfig
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


    cdef cppclass IRNGMixin:

        # Functions:

        IRNGConfig& useRNG()


    cdef cppclass ISequentialRuleModelAssemblageMixin:

        # Functions:

        void useSequentialRuleModelAssemblage()


    cdef cppclass IDefaultRuleMixin:

        # Functions:

        void useDefaultRule()


    cdef cppclass IGreedyTopDownRuleInductionMixin:

        # Functions:

        IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction()

        
    cdef cppclass IBeamSearchTopDownRuleInductionMixin:

        # Functions:

        IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction()


    cdef cppclass INoPostProcessorMixin:

        # Functions:

        void useNoPostProcessor()


    cdef cppclass INoFeatureBinningMixin:

        # Functions:

        void useNoFeatureBinning()
        

    cdef cppclass IEqualWidthFeatureBinningMixin:

        # Functions:

        IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning()


    cdef cppclass IEqualFrequencyFeatureBinningMixin:

        # Functions:

        IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning()


    cdef cppclass INoOutputSamplingMixin:

        # Functions:

        void useNoOutputSampling()


    cdef cppclass IRoundRobinOutputSamplingMixin:

        # Functions:

        void useRoundRobinOutputSampling()
    
    
    cdef cppclass IOutputSamplingWithoutReplacementMixin:

        # Functions:

        IOutputSamplingWithoutReplacementConfig& useOutputSamplingWithoutReplacement()


    cdef cppclass INoInstanceSamplingMixin:

        # Functions:

        void useNoInstanceSampling()


    cdef cppclass IInstanceSamplingWithoutReplacementMixin:

        # Functions:

        IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement()


    cdef cppclass IInstanceSamplingWithReplacementMixin:

        # Functions:

        IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement()


    cdef cppclass INoFeatureSamplingMixin:

        # Functions:

        void useNoFeatureSampling()


    cdef cppclass IFeatureSamplingWithoutReplacementMixin:

        # Functions:

        IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement()


    cdef cppclass INoPartitionSamplingMixin:

        # Functions:

        void useNoPartitionSampling()


    cdef cppclass IRandomBiPartitionSamplingMixin:

        # Functions:

        IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling()


    cdef cppclass INoRulePruningMixin:

        # Functions:

        void useNoRulePruning()


    cdef cppclass IIrepRulePruningMixin:

        # Functions:

        void useIrepRulePruning()


    cdef cppclass INoParallelRuleRefinementMixin:

        # Functions:

        void useNoParallelRuleRefinement()


    cdef cppclass IParallelRuleRefinementMixin:

        # Functions:

        IManualMultiThreadingConfig& useParallelRuleRefinement()

    
    cdef cppclass INoParallelStatisticUpdateMixin:

        # Functions:

        void useNoParallelStatisticUpdate()


    cdef cppclass IParallelStatisticUpdateMixin:

        # Functions:

        IManualMultiThreadingConfig& useParallelStatisticUpdate()


    cdef cppclass INoParallelPredictionMixin:

        # Functions:

        void useNoParallelPrediction()

        
    cdef cppclass IParallelPredictionMixin:

        # Functions:

        IManualMultiThreadingConfig& useParallelPrediction()


    cdef cppclass INoSizeStoppingCriterionMixin:

        # Functions:

        void useNoSizeStoppingCriterion()


    cdef cppclass ISizeStoppingCriterionMixin:

        # Functions:

        ISizeStoppingCriterionConfig& useSizeStoppingCriterion()


    cdef cppclass INoTimeStoppingCriterionMixin:

        # Functions:

        void useNoTimeStoppingCriterion()


    cdef cppclass ITimeStoppingCriterionMixin:

        # Functions:

        ITimeStoppingCriterionConfig& useTimeStoppingCriterion()


    cdef cppclass IPrePruningMixin:

        # Functions:

        IPrePruningConfig& useGlobalPrePruning()


    cdef cppclass INoGlobalPruningMixin:

        # Functions:

        void useNoGlobalPruning()


    cdef cppclass IPostPruningMixin:

        # Functions:

        IPostPruningConfig& useGlobalPostPruning()


    cdef cppclass INoSequentialPostOptimizationMixin:

        # Functions:

        void useNoSequentialPostOptimization()


    cdef cppclass ISequentialPostOptimizationMixin:

        # Functions:

        ISequentialPostOptimizationConfig& useSequentialPostOptimization()


    cdef cppclass INoMarginalProbabilityCalibrationMixin:

        # Functions:

        void useNoMarginalProbabilityCalibration()


    cdef cppclass INoJointProbabilityCalibrationMixin:

        # Functions:

        void useNoJointProbabilityCalibration()


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
