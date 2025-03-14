/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner.hpp"
#include "mlrl/common/learner_regression.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines the interface of the BOOMER algorithm for regression problems.
     */
    class MLRLBOOSTING_API IBoomerRegressor : virtual public IRegressionRuleLearner {
        public:

            /**
             * Defines the interface for configuring the BOOMER algorithm for classification problems.
             */
            class IConfig : virtual public IBoostedRuleLearnerMixin,
                            virtual public IAutomaticPartitionSamplingMixin,
                            virtual public IAutomaticFeatureBinningMixin,
                            virtual public IAutomaticParallelRuleRefinementMixin,
                            virtual public IAutomaticParallelStatisticUpdateMixin,
                            virtual public IConstantShrinkageMixin,
                            virtual public IL1RegularizationMixin,
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
                            virtual public IDecomposableSquaredErrorLossMixin,
                            virtual public INonDecomposableSquaredErrorLossMixin,
                            virtual public IOutputWiseScorePredictorMixin,
                            virtual public ISequentialRuleModelAssemblageMixin,
                            virtual public IGreedyTopDownRuleInductionMixin,
                            virtual public IBeamSearchTopDownRuleInductionMixin,
                            virtual public IEqualWidthFeatureBinningMixin,
                            virtual public IEqualFrequencyFeatureBinningMixin,
                            virtual public IRoundRobinOutputSamplingMixin,
                            virtual public IOutputSamplingWithoutReplacementMixin,
                            virtual public IInstanceSamplingWithoutReplacementMixin,
                            virtual public IInstanceSamplingWithReplacementMixin,
                            virtual public IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRandomBiPartitionSamplingMixin,
                            virtual public IIrepRulePruningMixin,
                            virtual public IParallelRuleRefinementMixin,
                            virtual public IParallelStatisticUpdateMixin,
                            virtual public IParallelPredictionMixin,
                            virtual public ISizeStoppingCriterionMixin,
                            virtual public ITimeStoppingCriterionMixin,
                            virtual public IPrePruningMixin,
                            virtual public IPostPruningMixin,
                            virtual public ISequentialPostOptimizationMixin {
                public:

                    virtual ~IConfig() override {}
            };

            virtual ~IBoomerRegressor() override {}
    };

    /**
     * Creates and returns a new object of type `IBoomerRegressor::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoomerRegressor::IConfig` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomerRegressor::IConfig> createBoomerRegressorConfig();

    /**
     * Creates and returns a new object of type `IBoomerRegressor`.
     *
     * @param configPtr     An unique pointer to an object of type `IBoomerRegressor::IConfig` that specifies the
     *                      configuration that should be used by the rule learner
     * @param sdotFunction  A function pointer to BLAS' SDOT routine
     * @param ddotFunction  A function pointer to BLAS' DDOT routine
     * @param sspmvFunction A function pointer to BLAS' SSPMV routine
     * @param dspmvFunction A function pointer to BLAS' DSPMV routine
     * @param ssysvFunction A function pointer to LAPACK'S SSYSV routine
     * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
     * @return              An unique pointer to an object of type `IBoomerRegressor` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomerRegressor> createBoomerRegressor(
      std::unique_ptr<IBoomerRegressor::IConfig> configPtr, Blas<float32>::DotFunction sdotFunction,
      Blas<float64>::DotFunction ddotFunction, Blas<float32>::SpmvFunction sspmvFunction,
      Blas<float64>::SpmvFunction dspmvFunction, Lapack<float32>::SysvFunction ssysvFunction,
      Lapack<float64>::SysvFunction dsysvFunction);

}
