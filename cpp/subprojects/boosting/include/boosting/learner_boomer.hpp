/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "boosting/learner.hpp"


namespace boosting {

    /**
     * Defines the interface of the BOOMER algorithm.
     */
    class MLRLBOOSTING_API IBoomer : virtual public IBoostingRuleLearner {

        public:

            /**
             * Defines the interface for configuring the BOOMER algorithm.
             */
            class IConfig : virtual public IBoostingRuleLearner::IConfig,
                            virtual public IBoostingRuleLearner::IShrinkageMixin,
                            virtual public IBoostingRuleLearner::IRegularizationMixin,
                            virtual public IBoostingRuleLearner::INoDefaultRuleMixin,
                            virtual public IBoostingRuleLearner::IPartialHeadMixin,
                            virtual public IBoostingRuleLearner::ISparseStatisticsMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseLogisticLossMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseSquaredErrorLossMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseSquaredHingeLossMixin,
                            virtual public IBoostingRuleLearner::ILabelBinningMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseClassificationPredictorMixin,
                            virtual public IBoostingRuleLearner::IMarginalizedProbabilityPredictorMixin,
                            virtual public IRuleLearner::IBeamSearchTopDownMixin,
                            virtual public IRuleLearner::IFeatureBinningMixin,
                            virtual public IRuleLearner::ILabelSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingMixin,
                            virtual public IRuleLearner::IPartitionSamplingMixin,
                            virtual public IRuleLearner::IPruningMixin,
                            virtual public IRuleLearner::IMultiThreadingMixin,
                            virtual public IRuleLearner::ISizeStoppingCriterionMixin,
                            virtual public IRuleLearner::ITimeStoppingCriterionMixin,
                            virtual public IRuleLearner::IEarlyStoppingCriterionMixin {

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learner to automatically decide whether a default rule should be induced or
                     * not.
                     */
                    virtual void useAutomaticDefaultRule() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of
                     * numerical feature values to bins should be used or not.
                     */
                    virtual void useAutomaticFeatureBinning() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel refinement of rules or not.
                     */
                    virtual void useAutomaticParallelRuleRefinement() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel update of statistics or not.
                     */
                    virtual void useAutomaticParallelStatisticUpdate() = 0;

                    /**
                     * Configures the rule learner to automatically decide for the type of rule heads that should be
                     * used.
                     */
                    virtual void useAutomaticHeads() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a dense or sparse representation of
                     * gradients and Hessians should be used.
                     */
                    virtual void useAutomaticStatistics() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of labels
                     * to bins should be used or not.
                     */
                    virtual void useAutomaticLabelBinning() = 0;

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting whether
                     * individual labels are relevant or irrelevant.
                     */
                    virtual void useAutomaticClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting probability
                     * estimates.
                     */
                    virtual void useAutomaticProbabilityPredictor() = 0;

            };

            virtual ~IBoomer() override { };

    };

    /**
     * The BOOMER algorithm.
     */
    class Boomer final : public AbstractBoostingRuleLearner, virtual public IBoomer {

        public:

            /**
             * Allows to configure the BOOMER algorithm.
             */
            class Config final : public AbstractBoostingRuleLearner::Config, virtual public IBoomer::IConfig {

                public:

                    Config();

                    /**
                     * @see `IRuleLearner::IBeamSearchTopDownMixin::useBeamSearchTopDownRuleInduction`
                     */
                    IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() override;

                    /**
                     * @see `IRuleLearner::IFeatureBinningMixin::useEqualWidthFeatureBinning`
                     */
                    IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning() override;

                    /**
                     * @see `IRuleLearner::IFeatureBinningMixin::useEqualFrequencyFeatureBinning`
                     */
                    IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning() override;

                    /**
                     * @see `IRuleLearner::ILabelSamplingMixin::useLabelSamplingWithoutReplacement`
                     */
                    ILabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement() override;

                    /**
                     * @see `IRuleLearner::IInstanceSamplingMixin::useInstanceSamplingWithReplacement`
                     */
                    IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement() override;

                    /**
                     * @see `IRuleLearner::IInstanceSamplingMixin::useInstanceSamplingWithoutReplacement`
                     */
                    IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement() override;

                    /**
                     * @see `IRuleLearner::IInstanceSamplingMixin::useLabelWiseStratifiedInstanceSampling`
                     */
                    ILabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling() override;

                    /**
                     * @see `IRuleLearner::IInstanceSamplingMixin::useExampleWiseStratifiedInstanceSampling`
                     */
                    IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() override;

                    /**
                     * @see `IRuleLearner::IFeatureSamplingMixin::useFeatureSamplingWithoutReplacement`
                     */
                    IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement() override;

                    /**
                     * @see `IRuleLearner::IPartitionSamplingMixin::useRandomBiPartitionSampling`
                     */
                    IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling() override;

                    /**
                     * @see `IRuleLearner::IPartitionSamplingMixin::useLabelWiseStratifiedBiPartitionSampling`
                     */
                    ILabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling() override;

                    /**
                     * @see `IRuleLearner::IPartitionSamplingMixin::useExampleWiseStratifiedBiPartitionSampling`
                     */
                    IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() override;

                    /**
                     * @see `IRuleLearner::IPruningMixin::useIrepPruning`
                     */
                    void useIrepPruning() override;

                    /**
                     * @see `IRuleLearner::IMultiThreadingMixin::useParallelRuleRefinement`
                     */
                    IManualMultiThreadingConfig& useParallelRuleRefinement() override;

                    /**
                     * @see `IRuleLearner::IMultiThreadingMixin::useParallelStatisticUpdate`
                     */
                    IManualMultiThreadingConfig& useParallelStatisticUpdate() override;

                    /**
                     * @see `IRuleLearner::IMultiThreadingMixin::useParallelPrediction`
                     */
                    IManualMultiThreadingConfig& useParallelPrediction() override;

                    /**
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;

                    /**
                     * @see `IRuleLearner::ITimeStoppingCriterionMixin::useTimeStoppingCriterion`
                     */
                    ITimeStoppingCriterionConfig& useTimeStoppingCriterion() override;

                    /**
                     * @see `IRuleLearner::IEarlyStoppingCriterionMixin::useEarlyStoppingCriterion`
                     */
                    IEarlyStoppingCriterionConfig& useEarlyStoppingCriterion() override;

                    /**
                     * @see `IBoostingRuleLearner::IShrinkageMixin::useConstantShrinkagePostProcessor`
                     */
                    IConstantShrinkageConfig& useConstantShrinkagePostProcessor() override;

                    /**
                     * @see `IBoostingRuleLearner::IRegularizationMixin::useL1Regularization`
                     */
                    IManualRegularizationConfig& useL1Regularization() override;

                    /**
                     * @see `IBoostingRuleLearner::IRegularizationMixin::useL2Regularization`
                     */
                    IManualRegularizationConfig& useL2Regularization() override;

                    /**
                     * @see `IBoostingRuleLearner::INoDefaultRuleMixin::useNoDefaultRule`
                     */
                    void useNoDefaultRule() override;

                    /**
                     * @see `IBoostingRuleLearner::IPartialHeadMixin::useFixedPartialHeads`
                     */
                    IFixedPartialHeadConfig& useFixedPartialHeads() override;

                    /**
                     * @see `IBoostingRuleLearner::IPartialHeadMixin::useDynamicPartialHeads`
                     */
                    IDynamicPartialHeadConfig& useDynamicPartialHeads() override;

                    /**
                     * @see `IBoostingRuleLearner::IPartialHeadMixin::useSingleLabelHeads`
                     */
                    void useSingleLabelHeads() override;

                    /**
                     * @see `IBoostingRuleLearner::ISparseStatisticsMixin::useSparseStatistics`
                     */
                    void useSparseStatistics() override;

                    /**
                     * @see `IBoostingRuleLearner::IExampleWiseLogisticLossMixin::useExampleWiseLogisticLoss`
                     */
                    void useExampleWiseLogisticLoss() override;

                    /**
                     * @see `IBoostingRuleLearner::ILabelWiseSquaredErrorLossMixin::useLabelWiseSquaredErrorLoss`
                     */
                    void useLabelWiseSquaredErrorLoss() override;

                    /**
                     * @see `IBoostingRuleLearner::ILabelWiseSquaredHingeLossMixin::useLabelWiseSquaredHingeLoss`
                     */
                    void useLabelWiseSquaredHingeLoss() override;

                    /**
                     * @see `IBoostingRuleLearner::ILabelBinningMixin::useEqualWidthLabelBinning`
                     */
                    IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() override;

                    /**
                     * @see `IBoostingRuleLearner::IExampleWiseClassificationPredictorMixin::useExampleWiseClassificationPredictor`
                     */
                    void useExampleWiseClassificationPredictor() override;

                    /**
                     * @see `IBoostingRuleLearner::IMarginalizedProbabilityPredictorMixin::useMarginalizedProbabilityPredictor`
                     */
                    void useMarginalizedProbabilityPredictor() override;

                    void useAutomaticDefaultRule() override;

                    void useAutomaticFeatureBinning() override;

                    void useAutomaticParallelRuleRefinement() override;

                    void useAutomaticParallelStatisticUpdate() override;

                    void useAutomaticHeads() override;

                    void useAutomaticStatistics() override;

                    void useAutomaticLabelBinning() override;

                    void useAutomaticClassificationPredictor() override;

                    void useAutomaticProbabilityPredictor() override;

            };

        private:

            std::unique_ptr<IBoomer::IConfig> configPtr_;

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
    #pragma warning ( pop )
#endif
