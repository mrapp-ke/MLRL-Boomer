/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"
#include "seco/heuristics/heuristic_accuracy.hpp"
#include "seco/heuristics/heuristic_f_measure.hpp"
#include "seco/heuristics/heuristic_laplace.hpp"
#include "seco/heuristics/heuristic_m_estimate.hpp"
#include "seco/heuristics/heuristic_precision.hpp"
#include "seco/heuristics/heuristic_recall.hpp"
#include "seco/heuristics/heuristic_wra.hpp"
#include "seco/lift_functions/lift_function_peak.hpp"
#include "seco/output/predictor_classification_label_wise.hpp"
#include "seco/stopping/stopping_criterion_coverage.hpp"


namespace seco {

    /**
     * Defines an interface for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class ISeCoRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of the
             * separate-and-conquer (SeCo) paradigm.
             */
            class IConfig : virtual public IRuleLearner::IConfig {

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
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IAccuracyConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IAccuracyConfig& useAccuracyHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasureHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `ILaplaceConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual ILaplaceConfig& useLaplaceHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimateHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IPrecisionConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IPrecisionConfig& usePrecisionHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IRecallConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual IRecallConfig& useRecallHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IWraConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual IWraConfig& useWraHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IAccuracyConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IAccuracyConfig& useAccuracyPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasurePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `ILaplaceConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual ILaplaceConfig& useLaplacePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimatePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IPrecisionConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IPrecisionConfig& usePrecisionPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IRecallConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual IRecallConfig& useRecallPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IWraConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual IWraConfig& useWraPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases until a certain
                     * number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
                     *
                     * @return A reference to an object of type `IPeakLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IPeakLiftFunctionConfig& usePeakLiftFunction() = 0;

                    /**
                     * Configures the rule learner to use predictor for predicting whether individual labels of given
                     * query examples are relevant or irrelevant by processing rules of an existing rule-based model in
                     * the order they have been learned. If a rule covers an example, its prediction is applied to each
                     * label individually.
                     *
                     * @return A reference to an object of type `ILabelWiseClassificationPredictorConfig` that allows
                     *         further configuration of the predictor for predicting whether individual labels of given
                     *         query examples are relevant or irrelevant
                     */
                    virtual ILabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor() = 0;

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

                    std::unique_ptr<IHeuristicConfig> heuristicConfigPtr_;

                    std::unique_ptr<IHeuristicConfig> pruningHeuristicConfigPtr_;

                    std::unique_ptr<ILiftFunctionConfig> liftFunctionConfigPtr_;

                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                    const CoverageStoppingCriterionConfig* getCoverageStoppingCriterionConfig() const override;

                    const IHeuristicConfig& getHeuristicConfig() const override;

                    const IHeuristicConfig& getPruningHeuristicConfig() const override;

                    const ILiftFunctionConfig& getLiftFunctionConfig() const override;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override;

                public:

                    Config();

                    ITopDownRuleInductionConfig& useTopDownRuleInduction() override;

                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;

                    void useNoCoverageStoppingCriterion() override;

                    ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() override;

                    IAccuracyConfig& useAccuracyHeuristic() override;

                    IFMeasureConfig& useFMeasureHeuristic() override;

                    ILaplaceConfig& useLaplaceHeuristic() override;

                    IMEstimateConfig& useMEstimateHeuristic() override;

                    IPrecisionConfig& usePrecisionHeuristic() override;

                    IRecallConfig& useRecallHeuristic() override;

                    IWraConfig& useWraHeuristic() override;

                    IAccuracyConfig& useAccuracyPruningHeuristic() override;

                    IFMeasureConfig& useFMeasurePruningHeuristic() override;

                    ILaplaceConfig& useLaplacePruningHeuristic() override;

                    IMEstimateConfig& useMEstimatePruningHeuristic() override;

                    IPrecisionConfig& usePrecisionPruningHeuristic() override;

                    IRecallConfig& useRecallPruningHeuristic() override;

                    IWraConfig& useWraPruningHeuristic() override;

                    IPeakLiftFunctionConfig& usePeakLiftFunction() override;

                    ILabelWiseClassificationPredictorConfig& useLabelWiseClassificationPredictor() override;

            };

        private:

            std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr_;

            std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const;

            std::unique_ptr<IHeuristicFactory> createPruningHeuristicFactory() const;

            std::unique_ptr<ILiftFunctionFactory> createLiftFunctionFactory() const;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const;

        protected:

            void createStoppingCriterionFactories(
                std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

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
    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `ISeCoRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `ISeCoRuleLearner` that has been created
     */
    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr);

}
