/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "common/learner.hpp"
#include "seco/heuristics/heuristic_f_measure.hpp"
#include "seco/heuristics/heuristic_m_estimate.hpp"
#include "seco/lift_functions/lift_function_kln.hpp"
#include "seco/lift_functions/lift_function_peak.hpp"
#include "seco/rule_evaluation/head_type.hpp"
#include "seco/stopping/stopping_criterion_coverage.hpp"


namespace seco {

    /**
     * Defines an interface for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class MLRLSECO_API ISeCoRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of the
             * separate-and-conquer (SeCo) paradigm.
             */
            class IConfig : virtual public IRuleLearner::IConfig,
                            virtual public IRuleLearner::IBeamSearchTopDownMixin,
                            virtual public IRuleLearner::IFeatureBinningMixin,
                            virtual public IRuleLearner::ILabelSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingMixin,
                            virtual public IRuleLearner::IPartitionSamplingMixin,
                            virtual public IRuleLearner::IPruningMixin,
                            virtual public IRuleLearner::IMultiThreadingMixin {

                friend class SeCoRuleLearner;

                private:

                    /**
                     * Returns the configuration of the stopping criterion that stops the induction of rules as soon as
                     * the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
                     *
                     * @return A pointer to an object of type `CoverageStoppingCriterionConfig` that specifies the
                     *         configuration of the stopping criterion that stops the induction of rules as soon as a
                     *         the sum of the weights of the uncovered labels is smaller or equal to a certain threshold
                     *         or a null pointer, if no such stopping criterion should be used
                     */
                    virtual const CoverageStoppingCriterionConfig* getCoverageStoppingCriterionConfig() const = 0;

                    /**
                     * Returns the configuration of the rule heads that should be induced by the rule learner.
                     *
                     * @return A reference to an object of type `IHeadConfig` that specifies the configuration of the
                     *         rule heads
                     */
                    virtual const IHeadConfig& getHeadConfig() const = 0;

                    /**
                     * Returns the configuration of the heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IHeuristicConfig` that specifies the configuration of
                     *         the heuristic for learning rules
                     */
                    virtual const IHeuristicConfig& getHeuristicConfig() const = 0;

                    /**
                     * Returns the configuration of the heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IHeuristicConfig` that specifies the configuration of
                     *         the heuristic for pruning rules
                     */
                    virtual const IHeuristicConfig& getPruningHeuristicConfig() const = 0;

                    /**
                     * Returns the configuration of the lift function that affects the quality of rules, depending on
                     * the number of labels for which they predict.
                     *
                     * @return A reference to an object of type `ILiftFunctionConfig` that specifies the configuration
                     *         of the lift function that affects the quality of rules, depending on the number of labels
                     *         for which they predict
                     */
                    virtual const ILiftFunctionConfig& getLiftFunctionConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts whether individual labels of given query
                     * examples are relevant or irrelevant.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts whether individual labels of given query
                     *         examples are relevant or irrelevant
                     */
                    virtual const IClassificationPredictorConfig& getClassificationPredictorConfig() const = 0;

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learner to not use any stopping criterion that stops the induction of rules
                     * as soon as the sum of the weights of the uncovered labels is smaller or equal to a certain
                     * threshold.
                     */
                    virtual void useNoCoverageStoppingCriterion() = 0;

                    /**
                     * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon
                     * as the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
                     *
                     * @return A reference to an object of type `ICoverageStoppingCriterionConfig` that allows further
                     *         configuration of the stopping criterion
                     */
                    virtual ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() = 0;

                    /**
                     * Configures the rule learner to induce rules with single-label heads that predict for a single
                     * label.
                     */
                    virtual void useSingleLabelHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels.
                     */
                    virtual void usePartialHeads() = 0;

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     */
                    virtual void useAccuracyHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasureHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     */
                    virtual void useLaplaceHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimateHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for learning rules.
                     */
                    virtual void usePrecisionHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     */
                    virtual void useRecallHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
                     */
                    virtual void useWraHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     */
                    virtual void useAccuracyPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasurePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     */
                    virtual void useLaplacePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimatePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for pruning rules.
                     */
                    virtual void usePrecisionPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     */
                    virtual void useRecallPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
                     */
                    virtual void useWraPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases until a certain
                     * number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
                     *
                     * @return A reference to an object of type `IPeakLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IPeakLiftFunctionConfig& usePeakLiftFunction() = 0;

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases according to the
                     * natural logarithm of the number of labels for which a rule predicts.
                     *
                     * @return A reference to an object of type `IKlnLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IKlnLiftFunctionConfig& useKlnLiftFunction() = 0;

                    /**
                     * Configures the rule learner to use predictor for predicting whether individual labels of given
                     * query examples are relevant or irrelevant by processing rules of an existing rule-based model in
                     * the order they have been learned. If a rule covers an example, its prediction is applied to each
                     * label individually.
                     */
                    virtual void useLabelWiseClassificationPredictor() = 0;

            };

            virtual ~ISeCoRuleLearner() override { };

    };

    /**
     * An implementation of the type `ISeCoRuleLearner`.
     */
    class SeCoRuleLearner final : public AbstractRuleLearner, virtual public ISeCoRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config final : public AbstractRuleLearner::Config, virtual public ISeCoRuleLearner::IConfig {

                private:

                    std::unique_ptr<CoverageStoppingCriterionConfig> coverageStoppingCriterionConfigPtr_;

                    std::unique_ptr<IHeadConfig> headConfigPtr_;

                    std::unique_ptr<IHeuristicConfig> heuristicConfigPtr_;

                    std::unique_ptr<IHeuristicConfig> pruningHeuristicConfigPtr_;

                    std::unique_ptr<ILiftFunctionConfig> liftFunctionConfigPtr_;

                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                    const CoverageStoppingCriterionConfig* getCoverageStoppingCriterionConfig() const override;

                    const IHeadConfig& getHeadConfig() const override;

                    const IHeuristicConfig& getHeuristicConfig() const override;

                    const IHeuristicConfig& getPruningHeuristicConfig() const override;

                    const ILiftFunctionConfig& getLiftFunctionConfig() const override;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override;

                public:

                    Config();

                    /**
                     * @see `IRuleLearner::IConfig::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override;

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
                     * @see `IRuleLearner::IConfig::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;

                    void useNoCoverageStoppingCriterion() override;

                    ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() override;

                    void useSingleLabelHeads() override;

                    void usePartialHeads() override;

                    void useAccuracyHeuristic() override;

                    IFMeasureConfig& useFMeasureHeuristic() override;

                    void useLaplaceHeuristic() override;

                    IMEstimateConfig& useMEstimateHeuristic() override;

                    void usePrecisionHeuristic() override;

                    void useRecallHeuristic() override;

                    void useWraHeuristic() override;

                    void useAccuracyPruningHeuristic() override;

                    IFMeasureConfig& useFMeasurePruningHeuristic() override;

                    void useLaplacePruningHeuristic() override;

                    IMEstimateConfig& useMEstimatePruningHeuristic() override;

                    void usePrecisionPruningHeuristic() override;

                    void useRecallPruningHeuristic() override;

                    void useWraPruningHeuristic() override;

                    IPeakLiftFunctionConfig& usePeakLiftFunction() override;

                    IKlnLiftFunctionConfig& useKlnLiftFunction() override;

                    void useLabelWiseClassificationPredictor() override;

            };

        private:

            std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr_;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const;

        protected:

            void createStoppingCriterionFactories(
                std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override;

            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        public:

            /**
             * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *                  configuration that should be used by the rule learner
             */
            SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr);

    };

    /**
     * Creates and returns a new object of type `ISeCoRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that has been created
     */
    MLRLSECO_API std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `ISeCoRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `ISeCoRuleLearner` that has been created
     */
    MLRLSECO_API std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(
        std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr);

}

#ifdef _WIN32
    #pragma warning ( pop )
#endif
