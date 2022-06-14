/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "boosting/learner.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/rule_evaluation/head_type_partial_dynamic.hpp"
#include "boosting/rule_evaluation/head_type_partial_fixed.hpp"


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
                            virtual public IRuleLearner::IMeasureStoppingCriterionMixin {

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learner to not induce a default rule.
                     */
                    virtual void useNoDefaultRule() = 0;

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
                     * Configures the rule learner to induce rules with partial heads that predict for a predefined
                     * number of labels.
                     *
                     * @return A reference to an object of type `IFixedPartialHeadConfig` that allows further
                     *         configuration of the rule heads
                     */
                    virtual IFixedPartialHeadConfig& useFixedPartialHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels that is determined dynamically. Only those labels for which the square of the
                     * predictive quality exceeds a certain threshold are included in a rule head.
                     *
                     * @return A reference to an object of type `IDynamicPartialHeadConfig` that allows further
                     *         configuration of the rule heads
                     */
                    virtual IDynamicPartialHeadConfig& useDynamicPartialHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with complete heads that predict for all available
                     * labels.
                     */
                    virtual void useCompleteHeads() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a dense or sparse representation of
                     * gradients and Hessians should be used.
                     */
                    virtual void useAutomaticStatistics() = 0;

                    /**
                     * Configures the rule learner to use a sparse representation of gradients and Hessians, if
                     * possible.
                     */
                    virtual void useSparseStatistics() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     */
                    virtual void useExampleWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredErrorLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredHingeLoss() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of labels
                     * to bins should be used or not.
                     */
                    virtual void useAutomaticLabelBinning() = 0;

                    /**
                     * Configures the rule learner to use a method for the assignment of labels to bins in a way such
                     * that each bin contains labels for which the predicted score is expected to belong to the same
                     * value range.
                     *
                     * @return A reference to an object of type `IEqualWidthLabelBinningConfig` that allows further
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by an existing rule-based model
                     * and comparing the aggregated score vector to the known label vectors according to a certain
                     * distance measure. The label vector that is closest to the aggregated score vector is finally
                     * predicted.
                     */
                    virtual void useExampleWiseClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting whether
                     * individual labels are relevant or irrelevant.
                     */
                    virtual void useAutomaticClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting probability estimates by summing up
                     * the scores that are provided by individual rules of an existing rule-based model and comparing
                     * the aggregated score vector to the known label vectors according to a certain distance measure.
                     * The probability for an individual label calculates as the sum of the distances that have been
                     * obtained for all label vectors, where the respective label is specified to be relevant, divided
                     * by the total sum of all distances.
                     */
                    virtual void useMarginalizedProbabilityPredictor() = 0;

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
                     * @see `IRuleLearner::IMeasureStoppingCriterionMixin::useMeasureStoppingCriterion`
                     */
                    IMeasureStoppingCriterionConfig& useMeasureStoppingCriterion() override;

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

                    void useNoDefaultRule() override;

                    void useAutomaticDefaultRule() override;

                    void useAutomaticFeatureBinning() override;

                    void useAutomaticParallelRuleRefinement() override;

                    void useAutomaticParallelStatisticUpdate() override;

                    void useAutomaticHeads() override;

                    IFixedPartialHeadConfig& useFixedPartialHeads() override;

                    IDynamicPartialHeadConfig& useDynamicPartialHeads() override;

                    void useCompleteHeads() override;

                    void useAutomaticStatistics() override;

                    void useSparseStatistics() override;

                    void useExampleWiseLogisticLoss() override;

                    void useLabelWiseSquaredErrorLoss() override;

                    void useLabelWiseSquaredHingeLoss() override;

                    void useAutomaticLabelBinning() override;

                    IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() override;

                    void useExampleWiseClassificationPredictor() override;

                    void useAutomaticClassificationPredictor() override;

                    void useMarginalizedProbabilityPredictor() override;

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
