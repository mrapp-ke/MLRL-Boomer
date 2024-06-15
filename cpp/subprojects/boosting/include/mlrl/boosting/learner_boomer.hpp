/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/learner.hpp"

namespace boosting {

    /**
     * Defines the interface of the BOOMER algorithm.
     */
    class MLRLBOOSTING_API IBoomer : virtual public IBoostedRuleLearner {
        public:

            /**
             * Defines the interface for configuring the BOOMER algorithm.
             */
            class IConfig : virtual public IBoostedRuleLearner::IConfig,
                            virtual public IBoostedRuleLearner::IAutomaticPartitionSamplingMixin,
                            virtual public IBoostedRuleLearner::IAutomaticFeatureBinningMixin,
                            virtual public IBoostedRuleLearner::IAutomaticParallelRuleRefinementMixin,
                            virtual public IBoostedRuleLearner::IAutomaticParallelStatisticUpdateMixin,
                            virtual public IBoostedRuleLearner::IConstantShrinkageMixin,
                            virtual public IBoostedRuleLearner::INoL1RegularizationMixin,
                            virtual public IBoostedRuleLearner::IL1RegularizationMixin,
                            virtual public IBoostedRuleLearner::INoL2RegularizationMixin,
                            virtual public IBoostedRuleLearner::IL2RegularizationMixin,
                            virtual public IBoostedRuleLearner::INoDefaultRuleMixin,
                            virtual public IBoostedRuleLearner::IAutomaticDefaultRuleMixin,
                            virtual public IBoostedRuleLearner::ICompleteHeadMixin,
                            virtual public IBoostedRuleLearner::IDynamicPartialHeadMixin,
                            virtual public IBoostedRuleLearner::IFixedPartialHeadMixin,
                            virtual public IBoostedRuleLearner::ISingleOutputHeadMixin,
                            virtual public IBoostedRuleLearner::IAutomaticHeadMixin,
                            virtual public IBoostedRuleLearner::IDenseStatisticsMixin,
                            virtual public IBoostedRuleLearner::ISparseStatisticsMixin,
                            virtual public IBoostedRuleLearner::IAutomaticStatisticsMixin,
                            virtual public IBoostedRuleLearner::IDecomposableLogisticLossMixin,
                            virtual public IBoostedRuleLearner::IDecomposableSquaredErrorLossMixin,
                            virtual public IBoostedRuleLearner::IDecomposableSquaredHingeLossMixin,
                            virtual public IBoostedRuleLearner::INonDecomposableLogisticLossMixin,
                            virtual public IBoostedRuleLearner::INonDecomposableSquaredErrorLossMixin,
                            virtual public IBoostedRuleLearner::INonDecomposableSquaredHingeLossMixin,
                            virtual public IBoostedRuleLearner::INoLabelBinningMixin,
                            virtual public IBoostedRuleLearner::IEqualWidthLabelBinningMixin,
                            virtual public IBoostedRuleLearner::IAutomaticLabelBinningMixin,
                            virtual public IBoostedRuleLearner::IIsotonicMarginalProbabilityCalibrationMixin,
                            virtual public IBoostedRuleLearner::IIsotonicJointProbabilityCalibrationMixin,
                            virtual public IBoostedRuleLearner::IOutputWiseBinaryPredictorMixin,
                            virtual public IBoostedRuleLearner::IExampleWiseBinaryPredictorMixin,
                            virtual public IBoostedRuleLearner::IGfmBinaryPredictorMixin,
                            virtual public IBoostedRuleLearner::IAutomaticBinaryPredictorMixin,
                            virtual public IBoostedRuleLearner::IOutputWiseScorePredictorMixin,
                            virtual public IBoostedRuleLearner::IOutputWiseProbabilityPredictorMixin,
                            virtual public IBoostedRuleLearner::IMarginalizedProbabilityPredictorMixin,
                            virtual public IBoostedRuleLearner::IAutomaticProbabilityPredictorMixin,
                            virtual public IRuleLearner::ISequentialRuleModelAssemblageMixin,
                            virtual public IRuleLearner::IDefaultRuleMixin,
                            virtual public IRuleLearner::IGreedyTopDownRuleInductionMixin,
                            virtual public IRuleLearner::IBeamSearchTopDownRuleInductionMixin,
                            virtual public IRuleLearner::INoPostProcessorMixin,
                            virtual public IRuleLearner::INoFeatureBinningMixin,
                            virtual public IRuleLearner::IEqualWidthFeatureBinningMixin,
                            virtual public IRuleLearner::IEqualFrequencyFeatureBinningMixin,
                            virtual public IRuleLearner::INoOutputSamplingMixin,
                            virtual public IRuleLearner::IRoundRobinOutputSamplingMixin,
                            virtual public IRuleLearner::IOutputSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoInstanceSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithReplacementMixin,
                            virtual public IRuleLearner::IOutputWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::INoFeatureSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoPartitionSamplingMixin,
                            virtual public IRuleLearner::IRandomBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IOutputWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::INoRulePruningMixin,
                            virtual public IRuleLearner::IIrepRulePruningMixin,
                            virtual public IRuleLearner::INoParallelRuleRefinementMixin,
                            virtual public IRuleLearner::IParallelRuleRefinementMixin,
                            virtual public IRuleLearner::INoParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::IParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::INoParallelPredictionMixin,
                            virtual public IRuleLearner::IParallelPredictionMixin,
                            virtual public IRuleLearner::INoSizeStoppingCriterionMixin,
                            virtual public IRuleLearner::ISizeStoppingCriterionMixin,
                            virtual public IRuleLearner::INoTimeStoppingCriterionMixin,
                            virtual public IRuleLearner::ITimeStoppingCriterionMixin,
                            virtual public IRuleLearner::IPrePruningMixin,
                            virtual public IRuleLearner::INoGlobalPruningMixin,
                            virtual public IRuleLearner::IPostPruningMixin,
                            virtual public IRuleLearner::INoSequentialPostOptimizationMixin,
                            virtual public IRuleLearner::ISequentialPostOptimizationMixin,
                            virtual public IRuleLearner::INoMarginalProbabilityCalibrationMixin,
                            virtual public IRuleLearner::INoJointProbabilityCalibrationMixin {
                public:

                    virtual ~IConfig() override {}
            };

            virtual ~IBoomer() override {}
    };

    /**
     * The BOOMER algorithm.
     */
    class Boomer final : public AbstractBoostingRuleLearner,
                         virtual public IBoomer {
        public:

            /**
             * Allows to configure the BOOMER algorithm.
             */
            class Config final : public AbstractBoostingRuleLearner::Config,
                                 virtual public IBoomer::IConfig {
                public:

                    Config();

                    /**
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;
            };

        private:

            const std::unique_ptr<IBoomer::IConfig> configPtr_;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoomer::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);
    };

    /**
     * Creates and returns a new object of type `IBoomer::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoomer::IConfig` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomer::IConfig> createBoomerConfig();

    /**
     * Creates and returns a new object of type `IBoomer`.
     *
     * @param configPtr     An unique pointer to an object of type `IBoomer::IConfig` that specifies the configuration
     *                      that should be used by the rule learner
     * @param ddotFunction  A function pointer to BLAS' DDOT routine
     * @param dspmvFunction A function pointer to BLAS' DSPMV routine
     * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
     * @return              An unique pointer to an object of type `IBoomer` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr,
                                                           Blas::DdotFunction ddotFunction,
                                                           Blas::DspmvFunction dspmvFunction,
                                                           Lapack::DsysvFunction dsysvFunction);

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
