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
                     * Returns the configuration of the heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IHeuristic` that specifies the configuration of the
                     *         heuristic for learning rules
                     */
                    virtual const IHeuristicConfig& getHeuristicConfig() const = 0;

                    /**
                     * Returns the configuration of the heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IHeuristic` that specifies the configuration of the
                     *         heuristic for pruning rules
                     */
                    virtual const IHeuristicConfig& getPruningHeuristicConfig() const = 0;

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `AccuracyConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual AccuracyConfig& useAccuracyHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `FMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual FMeasureConfig& useFMeasureHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `LaplaceConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual LaplaceConfig& useLaplaceHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `MEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual MEstimateConfig& useMEstimateHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `PrecisionConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual PrecisionConfig& usePrecisionHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `RecallConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual RecallConfig& useRecallHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `WraConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual WraConfig& useWraHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `AccuracyConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual AccuracyConfig& useAccuracyPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `FMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual FMeasureConfig& useFMeasurePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `LaplaceConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual LaplaceConfig& useLaplacePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `MEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual MEstimateConfig& useMEstimatePruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Precision" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `PrecisionConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual PrecisionConfig& usePrecisionPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `RecallConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual RecallConfig& useRecallPruningHeuristic() = 0;

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `WraConfig` that allows further configuration of the
                     *         heuristic
                     */
                    virtual WraConfig& useWraPruningHeuristic() = 0;

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
            class Config : public AbstractRuleLearner::Config, virtual public ISeCoRuleLearner::IConfig {

                private:

                    std::unique_ptr<IHeuristicConfig> heuristicConfigPtr_;

                    std::unique_ptr<IHeuristicConfig> pruningHeuristicConfigPtr_;

                    const IHeuristicConfig& getHeuristicConfig() const override;

                    const IHeuristicConfig& getPruningHeuristicConfig() const override;

                public:

                    Config();

                    AccuracyConfig& useAccuracyHeuristic() override;

                    FMeasureConfig& useFMeasureHeuristic() override;

                    LaplaceConfig& useLaplaceHeuristic() override;

                    MEstimateConfig& useMEstimateHeuristic() override;

                    PrecisionConfig& usePrecisionHeuristic() override;

                    RecallConfig& useRecallHeuristic() override;

                    WraConfig& useWraHeuristic() override;

                    AccuracyConfig& useAccuracyPruningHeuristic() override;

                    FMeasureConfig& useFMeasurePruningHeuristic() override;

                    LaplaceConfig& useLaplacePruningHeuristic() override;

                    MEstimateConfig& useMEstimatePruningHeuristic() override;

                    PrecisionConfig& usePrecisionPruningHeuristic() override;

                    RecallConfig& useRecallPruningHeuristic() override;

                    WraConfig& useWraPruningHeuristic() override;

            };

        protected:

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
