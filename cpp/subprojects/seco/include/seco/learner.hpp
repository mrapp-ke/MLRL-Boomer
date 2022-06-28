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
            class IConfig : virtual public IRuleLearner::IConfig {

                friend class AbstractSeCoRuleLearner;

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
                     * Configures the rule learner to induce rules with single-label heads that predict for a single
                     * label.
                     */
                    virtual void useSingleLabelHeads() = 0;

                    /**
                     * Configures the rule learner to not use a lift function.
                     */
                    virtual void useNoLiftFunction() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for learning rules.
                     */
                    virtual void usePrecisionHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for pruning rules.
                     */
                    virtual void usePrecisionPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use predictor for predicting whether individual labels of given
                     * query examples are relevant or irrelevant by processing rules of an existing rule-based model in
                     * the order they have been learned. If a rule covers an example, its prediction is applied to each
                     * label individually.
                     */
                    virtual void useLabelWiseClassificationPredictor() = 0;

            };

            virtual ~ISeCoRuleLearner() override { };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion
             * that stops the induction of rules as soon as the sum of the weights of the uncovered labels is smaller or
             * equal to a certain threshold.
             */
            class ICoverageStoppingCriterionMixin {

                public:

                    virtual ~ICoverageStoppingCriterionMixin() { };

                    /**
                     * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon
                     * as the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
                     *
                     * @return A reference to an object of type `ICoverageStoppingCriterionConfig` that allows further
                     *         configuration of the stopping criterion
                     */
                    virtual ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads.
             */
            class IPartialHeadMixin {

                public:

                    virtual ~IPartialHeadMixin() { };

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels.
                     */
                    virtual void usePartialHeads() = 0;

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

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Accuracy"
             * heuristic for learning or pruning rules.
             */
            class IAccuracyMixin {

                public:

                    virtual ~IAccuracyMixin() { };

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     */
                    virtual void useAccuracyHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     */
                    virtual void useAccuracyPruningHeuristic() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "F-Measure"
             * heuristic for learning or pruning rules.
             */
            class IFMeasureMixin {

                public:

                    virtual ~IFMeasureMixin() { };

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasureHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasurePruningHeuristic() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "M-Estimate"
             * heuristic for learning or pruning rules.
             */
            class IMEstimateMixin {

                public:

                    virtual ~IMEstimateMixin() { };

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimateHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimatePruningHeuristic() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Laplace"
             * heuristic for learning or pruning rules.
             */
            class ILaplaceMixin {

                public:

                    virtual ~ILaplaceMixin() { };

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     */
                    virtual void useLaplaceHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     */
                    virtual void useLaplacePruningHeuristic() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Recall" heuristic
             * for learning or pruning rules.
             */
            class IRecallMixin {

                public:

                    virtual ~IRecallMixin() { };

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     */
                    virtual void useRecallHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     */
                    virtual void useRecallPruningHeuristic() = 0;

            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Weighted Relative
             * Accuracy" heuristic for learning or pruning rules.
             */
            class IWraMixin {

                public:

                    virtual ~IWraMixin() { };

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
                     */
                    virtual void useWraHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
                     */
                    virtual void useWraPruningHeuristic() = 0;

            };

    };

    /**
     * An abstract base class for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class AbstractSeCoRuleLearner : public AbstractRuleLearner, virtual public ISeCoRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config : public AbstractRuleLearner::Config, virtual public ISeCoRuleLearner::IConfig {

                protected:

                    /**
                     * An unique pointer that stores the configuration of the stopping criterion that stops the
                     * induction of rules as soon as the sum of the weights of the uncovered labels is smaller or equal
                     * to a certain threshold.
                     */
                    std::unique_ptr<CoverageStoppingCriterionConfig> coverageStoppingCriterionConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the rule heads.
                     */
                    std::unique_ptr<IHeadConfig> headConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the heuristic that is used for learning rules.
                     */
                    std::unique_ptr<IHeuristicConfig> heuristicConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the heuristic that is used for pruning rules.
                     */
                    std::unique_ptr<IHeuristicConfig> pruningHeuristicConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the lift function that affects the quality of
                     * rules, depending on the number of labels for which they predict.
                     */
                    std::unique_ptr<ILiftFunctionConfig> liftFunctionConfigPtr_;

                    /**
                     * An unique pointer that stores the configuration of the predictor that is used to predict binary
                     * labels.
                     */
                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                private:

                    const CoverageStoppingCriterionConfig* getCoverageStoppingCriterionConfig() const override final;

                    const IHeadConfig& getHeadConfig() const override final;

                    const IHeuristicConfig& getHeuristicConfig() const override final;

                    const IHeuristicConfig& getPruningHeuristicConfig() const override final;

                    const ILiftFunctionConfig& getLiftFunctionConfig() const override final;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override final;

                public:

                    Config();

                    /**
                     * @see `IRuleLearner::IConfig::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override;

                    void useNoCoverageStoppingCriterion() override;

                    void useSingleLabelHeads() override;

                    void useNoLiftFunction() override;

                    void usePrecisionHeuristic() override;

                    void usePrecisionPruningHeuristic() override;

                    void useLabelWiseClassificationPredictor() override;

            };

        private:

            ISeCoRuleLearner::IConfig& config_;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const;

        protected:

            /**
             * @see `AbstractRuleLearner::createStoppingCriterionFactories`
             */
            void createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const override;

            /**
             * @see `AbstractRuleLearner::createStatisticsProviderFactory`
             */
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override;

            /**
             * @see `AbstractRuleLearner::createLabelSpaceInfo`
             */
            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createClassificationPredictorFactory`
             */
            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        public:

            /**
             * @param config A reference to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *               configuration that should be used by the rule learner
             */
            AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config);

    };

}

#ifdef _WIN32
    #pragma warning ( pop )
#endif
