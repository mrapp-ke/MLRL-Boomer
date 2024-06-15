/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/learner.hpp"
#include "mlrl/seco/model/decision_list_builder.hpp"
#include "mlrl/seco/rule_evaluation/rule_compare_function.hpp"

namespace seco {

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

                public:

                    Config()
                        : AbstractRuleLearner::Config(SECO_RULE_COMPARE_FUNCTION),
                          headConfigPtr_(
                            std::make_unique<SingleOutputHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_)),
                          heuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                          pruningHeuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                          liftFunctionConfigPtr_(std::make_unique<NoLiftFunctionConfig>()) {}

                    std::unique_ptr<CoverageStoppingCriterionConfig>& getCoverageStoppingCriterionConfigPtr()
                      override final {
                        return coverageStoppingCriterionConfigPtr_;
                    }

                    std::unique_ptr<IHeadConfig>& getHeadConfigPtr() override final {
                        return headConfigPtr_;
                    }

                    std::unique_ptr<IHeuristicConfig>& getHeuristicConfigPtr() override final {
                        return heuristicConfigPtr_;
                    }

                    std::unique_ptr<IHeuristicConfig>& getPruningHeuristicConfigPtr() override final {
                        return pruningHeuristicConfigPtr_;
                    }

                    std::unique_ptr<ILiftFunctionConfig>& getLiftFunctionConfigPtr() override final {
                        return liftFunctionConfigPtr_;
                    }
            };

        private:

            ISeCoRuleLearner::IConfig& config_;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const {
                std::unique_ptr<CoverageStoppingCriterionConfig>& configPtr =
                  config_.getCoverageStoppingCriterionConfigPtr();
                return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
            }

        protected:

            /**
             * @see `AbstractRuleLearner::createStoppingCriterionFactories`
             */
            void createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const override {
                AbstractRuleLearner::createStoppingCriterionFactories(factory);
                std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
                  this->createCoverageStoppingCriterionFactory();

                if (stoppingCriterionFactory) {
                    factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
                }
            }

            /**
             * @see `AbstractRuleLearner::createStatisticsProviderFactory`
             */
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return config_.getHeadConfigPtr()->createStatisticsProviderFactory(labelMatrix);
            }

            /**
             * @see `AbstractRuleLearner::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override {
                return std::make_unique<DecisionListBuilderFactory>();
            }

            /**
             * @see `AbstractRuleLearner::createSparseBinaryPredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparseBinaryPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
                return config_.getBinaryPredictorConfigPtr()->createSparsePredictorFactory(featureMatrix, numLabels);
            }

        public:

            /**
             * @param config A reference to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *               configuration that should be used by the rule learner
             */
            AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config) : AbstractRuleLearner(config), config_(config) {}
    };

}
