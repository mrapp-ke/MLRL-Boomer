/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner_classification.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Defines the interface of the BOOMER algorithm for classification problems.
     */
    class MLRLBOOSTING_API IBoomerClassifier : virtual public IClassificationRuleLearner {
        public:

            /**
             * Defines the interface for configuring the BOOMER algorithm for classification problems.
             */
            class IConfig : virtual public IBoostedRuleLearnerConfig,
                            virtual public IAutomaticPartitionSamplingMixin,
                            virtual public IAutomaticFeatureBinningMixin,
                            virtual public IAutomaticParallelRuleRefinementMixin,
                            virtual public IAutomaticParallelStatisticUpdateMixin,
                            virtual public IConstantShrinkageMixin,
                            virtual public INoL1RegularizationMixin,
                            virtual public IL1RegularizationMixin,
                            virtual public INoL2RegularizationMixin,
                            virtual public IL2RegularizationMixin,
                            virtual public INoDefaultRuleMixin,
                            virtual public IAutomaticDefaultRuleMixin,
                            virtual public ICompleteHeadMixin,
                            virtual public IDynamicPartialHeadMixin,
                            virtual public IFixedPartialHeadMixin,
                            virtual public ISingleOutputHeadMixin,
                            virtual public IAutomaticHeadMixin,
                            virtual public IDenseStatisticsMixin,
                            virtual public ISparseStatisticsMixin,
                            virtual public IAutomaticStatisticsMixin,
                            virtual public IDecomposableLogisticLossMixin,
                            virtual public IDecomposableSquaredErrorLossMixin,
                            virtual public IDecomposableSquaredHingeLossMixin,
                            virtual public INonDecomposableLogisticLossMixin,
                            virtual public INonDecomposableSquaredErrorLossMixin,
                            virtual public INonDecomposableSquaredHingeLossMixin,
                            virtual public INoLabelBinningMixin,
                            virtual public IEqualWidthLabelBinningMixin,
                            virtual public IAutomaticLabelBinningMixin,
                            virtual public IIsotonicMarginalProbabilityCalibrationMixin,
                            virtual public IIsotonicJointProbabilityCalibrationMixin,
                            virtual public IOutputWiseBinaryPredictorMixin,
                            virtual public IExampleWiseBinaryPredictorMixin,
                            virtual public IGfmBinaryPredictorMixin,
                            virtual public IAutomaticBinaryPredictorMixin,
                            virtual public IOutputWiseScorePredictorMixin,
                            virtual public IOutputWiseProbabilityPredictorMixin,
                            virtual public IMarginalizedProbabilityPredictorMixin,
                            virtual public IAutomaticProbabilityPredictorMixin,
                            virtual public ISequentialRuleModelAssemblageMixin,
                            virtual public IDefaultRuleMixin,
                            virtual public IGreedyTopDownRuleInductionMixin,
                            virtual public IBeamSearchTopDownRuleInductionMixin,
                            virtual public INoPostProcessorMixin,
                            virtual public INoFeatureBinningMixin,
                            virtual public IEqualWidthFeatureBinningMixin,
                            virtual public IEqualFrequencyFeatureBinningMixin,
                            virtual public INoOutputSamplingMixin,
                            virtual public IRoundRobinOutputSamplingMixin,
                            virtual public IOutputSamplingWithoutReplacementMixin,
                            virtual public INoInstanceSamplingMixin,
                            virtual public IInstanceSamplingWithoutReplacementMixin,
                            virtual public IInstanceSamplingWithReplacementMixin,
                            virtual public IOutputWiseStratifiedInstanceSamplingMixin,
                            virtual public IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public INoFeatureSamplingMixin,
                            virtual public IFeatureSamplingWithoutReplacementMixin,
                            virtual public INoPartitionSamplingMixin,
                            virtual public IRandomBiPartitionSamplingMixin,
                            virtual public IOutputWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public INoRulePruningMixin,
                            virtual public IIrepRulePruningMixin,
                            virtual public INoParallelRuleRefinementMixin,
                            virtual public IParallelRuleRefinementMixin,
                            virtual public INoParallelStatisticUpdateMixin,
                            virtual public IParallelStatisticUpdateMixin,
                            virtual public INoParallelPredictionMixin,
                            virtual public IParallelPredictionMixin,
                            virtual public INoSizeStoppingCriterionMixin,
                            virtual public ISizeStoppingCriterionMixin,
                            virtual public INoTimeStoppingCriterionMixin,
                            virtual public ITimeStoppingCriterionMixin,
                            virtual public IPrePruningMixin,
                            virtual public INoGlobalPruningMixin,
                            virtual public IPostPruningMixin,
                            virtual public INoSequentialPostOptimizationMixin,
                            virtual public ISequentialPostOptimizationMixin,
                            virtual public INoMarginalProbabilityCalibrationMixin,
                            virtual public INoJointProbabilityCalibrationMixin {
                public:

                    virtual ~IConfig() override {}
            };

            virtual ~IBoomerClassifier() override {}
    };

    /**
     * Creates and returns a new object of type `IBoomerClassifier::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoomerClassifier::IConfig` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomerClassifier::IConfig> createBoomerClassifierConfig();

    /**
     * Creates and returns a new object of type `IBoomerClassifier`.
     *
     * @param configPtr     An unique pointer to an object of type `IBoomerClassifier::IConfig` that specifies the
     *                      configuration that should be used by the rule learner
     * @param ddotFunction  A function pointer to BLAS' DDOT routine
     * @param dspmvFunction A function pointer to BLAS' DSPMV routine
     * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
     * @return              An unique pointer to an object of type `IBoomerClassifier` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomerClassifier> createBoomerClassifier(
      std::unique_ptr<IBoomerClassifier::IConfig> configPtr, Blas::DdotFunction ddotFunction,
      Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);

}
