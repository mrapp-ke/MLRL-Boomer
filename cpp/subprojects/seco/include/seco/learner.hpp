/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "common/learner.hpp"
#include "seco/heuristics/heuristic_accuracy.hpp"
#include "seco/heuristics/heuristic_f_measure.hpp"
#include "seco/heuristics/heuristic_laplace.hpp"
#include "seco/heuristics/heuristic_m_estimate.hpp"
#include "seco/heuristics/heuristic_recall.hpp"
#include "seco/heuristics/heuristic_wra.hpp"
#include "seco/lift_functions/lift_function_kln.hpp"
#include "seco/lift_functions/lift_function_peak.hpp"
#include "seco/rule_evaluation/head_type_partial.hpp"
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

                protected:

                    /**
                     * Returns an unique pointer to the configuration of the stopping criterion that stops the induction
                     * of rules as soon as the sum of the weights of the uncovered labels is smaller or equal to a
                     * certain threshold.
                     *
                     * @return A reference to an unique pointer of type `CoverageStoppingCriterionConfig` that stores
                     *         the configuration of the stopping criterion that stops the induction of rules as soon as
                     *         the sum of the weights of the uncovered labels is smaller or equal to a certain threshold
                     *         or a null pointer, if no such stopping criterion should be used
                     */
                    virtual std::unique_ptr<CoverageStoppingCriterionConfig>&
                      getCoverageStoppingCriterionConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the rule heads that should be induced by the
                     * rule learner.
                     *
                     * @return A reference to an unique pointer of type `IHeadConfig` that stores the configuration of
                     *         the rule heads
                     */
                    virtual std::unique_ptr<IHeadConfig>& getHeadConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the heuristic for learning rules.
                     *
                     * @return A reference to an unique pointer of type `IHeuristicConfig` that stores the configuration
                     *         of the heuristic for learning rules
                     */
                    virtual std::unique_ptr<IHeuristicConfig>& getHeuristicConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the heuristic for pruning rules.
                     *
                     * @return A reference to an unique pointer of type `IHeuristicConfig` that stores the configuration
                     *         of the heuristic for pruning rules
                     */
                    virtual std::unique_ptr<IHeuristicConfig>& getPruningHeuristicConfigPtr() = 0;

                    /**
                     * Returns an unique pointer to the configuration of the lift function that affects the quality of
                     * rules, depending on the number of labels for which they predict.
                     *
                     * @return A reference to an unique pointer of type `ILiftFunctionConfig` that stores the
                     *         configuration of the lift function that affects the quality of rules, depending on the
                     *         number of labels for which they predict
                     */
                    virtual std::unique_ptr<ILiftFunctionConfig>& getLiftFunctionConfigPtr() = 0;

                public:

                    virtual ~IConfig() override {};

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
                    virtual void useLabelWiseBinaryPredictor() = 0;
            };

            virtual ~ISeCoRuleLearner() override {};

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion
             * that stops the induction of rules as soon as the sum of the weights of the uncovered labels is smaller or
             * equal to a certain threshold.
             */
            class ICoverageStoppingCriterionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ICoverageStoppingCriterionMixin() override {};

                    /**
                     * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon
                     * as the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
                     *
                     * @return A reference to an object of type `ICoverageStoppingCriterionConfig` that allows further
                     *         configuration of the stopping criterion
                     */
                    virtual ICoverageStoppingCriterionConfig& useCoverageStoppingCriterion() {
                        std::unique_ptr<CoverageStoppingCriterionConfig>& coverageStoppingCriterionConfigPtr =
                          this->getCoverageStoppingCriterionConfigPtr();
                        std::unique_ptr<CoverageStoppingCriterionConfig> ptr =
                          std::make_unique<CoverageStoppingCriterionConfig>();
                        ICoverageStoppingCriterionConfig& ref = *ptr;
                        coverageStoppingCriterionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to induce rules with partial
             * heads.
             */
            class IPartialHeadMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPartialHeadMixin() override {};

                    /**
                     * Configures the rule learner to induce rules with partial heads that predict for a subset of the
                     * available labels.
                     */
                    virtual void usePartialHeads() {
                        std::unique_ptr<IHeadConfig>& headConfigPtr = this->getHeadConfigPtr();
                        headConfigPtr = std::make_unique<PartialHeadConfig>(this->getHeuristicConfigPtr(),
                                                                            this->getPruningHeuristicConfigPtr(),
                                                                            this->getLiftFunctionConfigPtr());
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a lift function that
             * monotonously increases until a certain number of labels, where the maximum lift is reached, and
             * monotonously decreases afterwards.
             */
            class IPeakLiftFunctionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IPeakLiftFunctionMixin() override {};

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases until a certain
                     * number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
                     *
                     * @return A reference to an object of type `IPeakLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IPeakLiftFunctionConfig& usePeakLiftFunction() {
                        std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr = this->getLiftFunctionConfigPtr();
                        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
                        IPeakLiftFunctionConfig& ref = *ptr;
                        liftFunctionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use a lift function that
             * monotonously increases according to the natural logarithm of the number of labels for which a rule
             * predicts.
             */
            class IKlnLiftFunctionMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IKlnLiftFunctionMixin() override {};

                    /**
                     * Configures the rule learner to use a lift function that monotonously increases according to the
                     * natural logarithm of the number of labels for which a rule predicts.
                     *
                     * @return A reference to an object of type `IKlnLiftFunctionConfig` that allows further
                     *         configuration of the lift function
                     */
                    virtual IKlnLiftFunctionConfig& useKlnLiftFunction() {
                        std::unique_ptr<ILiftFunctionConfig>& liftFunctionConfigPtr = this->getLiftFunctionConfigPtr();
                        std::unique_ptr<KlnLiftFunctionConfig> ptr = std::make_unique<KlnLiftFunctionConfig>();
                        IKlnLiftFunctionConfig& ref = *ptr;
                        liftFunctionConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Accuracy"
             * heuristic for learning or pruning rules.
             */
            class IAccuracyMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IAccuracyMixin() override {};

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for learning rules.
                     */
                    virtual void useAccuracyHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<AccuracyConfig>();
                    }

                    /**
                     * Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
                     */
                    virtual void useAccuracyPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<AccuracyConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "F-Measure"
             * heuristic for learning or pruning rules.
             */
            class IFMeasureMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IFMeasureMixin() override {};

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasureHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
                        IFMeasureConfig& ref = *ptr;
                        heuristicConfigPtr = std::move(ptr);
                        return ref;
                    }

                    /**
                     * Configures the rule learner to use the "F-Measure" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IFMeasureConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IFMeasureConfig& useFMeasurePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
                        IFMeasureConfig& ref = *ptr;
                        pruningHeuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "M-Estimate"
             * heuristic for learning or pruning rules.
             */
            class IMEstimateMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IMEstimateMixin() override {};

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for learning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimateHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
                        IMEstimateConfig& ref = *ptr;
                        heuristicConfigPtr = std::move(ptr);
                        return ref;
                    }

                    /**
                     * Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.
                     *
                     * @return A reference to an object of type `IMEstimateConfig` that allows further configuration of
                     *         the heuristic
                     */
                    virtual IMEstimateConfig& useMEstimatePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
                        IMEstimateConfig& ref = *ptr;
                        pruningHeuristicConfigPtr = std::move(ptr);
                        return ref;
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Laplace"
             * heuristic for learning or pruning rules.
             */
            class ILaplaceMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~ILaplaceMixin() override {};

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for learning rules.
                     */
                    virtual void useLaplaceHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<LaplaceConfig>();
                    }

                    /**
                     * Configures the rule learner to use the "Laplace" heuristic for pruning rules.
                     */
                    virtual void useLaplacePruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<LaplaceConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Recall" heuristic
             * for learning or pruning rules.
             */
            class IRecallMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IRecallMixin() override {};

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for learning rules.
                     */
                    virtual void useRecallHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<RecallConfig>();
                    }

                    /**
                     * Configures the rule learner to use the "Recall" heuristic for pruning rules.
                     */
                    virtual void useRecallPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<RecallConfig>();
                    }
            };

            /**
             * Defines an interface for all classes that allow to configure a rule learner to use the "Weighted Relative
             * Accuracy" heuristic for learning or pruning rules.
             */
            class IWraMixin : virtual public ISeCoRuleLearner::IConfig {
                public:

                    virtual ~IWraMixin() override {};

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
                     */
                    virtual void useWraHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& heuristicConfigPtr = this->getHeuristicConfigPtr();
                        heuristicConfigPtr = std::make_unique<WraConfig>();
                    }

                    /**
                     * Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
                     */
                    virtual void useWraPruningHeuristic() {
                        std::unique_ptr<IHeuristicConfig>& pruningHeuristicConfigPtr =
                          this->getPruningHeuristicConfigPtr();
                        pruningHeuristicConfigPtr = std::make_unique<WraConfig>();
                    }
            };
    };

    /**
     * An abstract base class for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class AbstractSeCoRuleLearner : public AbstractRuleLearner,
                                    virtual public ISeCoRuleLearner {
        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config : public AbstractRuleLearner::Config,
                           virtual public ISeCoRuleLearner::IConfig {
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

                private:

                    std::unique_ptr<CoverageStoppingCriterionConfig>& getCoverageStoppingCriterionConfigPtr()
                      override final;

                    std::unique_ptr<IHeadConfig>& getHeadConfigPtr() override final;

                    std::unique_ptr<IHeuristicConfig>& getHeuristicConfigPtr() override final;

                    std::unique_ptr<IHeuristicConfig>& getPruningHeuristicConfigPtr() override final;

                    std::unique_ptr<ILiftFunctionConfig>& getLiftFunctionConfigPtr() override final;

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

                    void useLabelWiseBinaryPredictor() override;
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
             * @see `AbstractRuleLearner::createSparseBinaryPredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparseBinaryPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        public:

            /**
             * @param config A reference to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *               configuration that should be used by the rule learner
             */
            AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config);
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
