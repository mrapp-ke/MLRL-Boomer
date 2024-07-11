/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/common/learner_common.hpp"
#include "mlrl/seco/learner.hpp"
#include "mlrl/seco/model/decision_list_builder.hpp"
#include "mlrl/seco/rule_evaluation/rule_compare_function.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * Allows to configure the individual modules of a rule learner that makes use of the separate-and-conquer (SeCo)
     * paradigm, depending on an `ISeCoRuleLearnerConfig`.
     */
    class SeCoRuleLearnerConfigurator final : public RuleLearnerConfigurator {
        private:

            std::unique_ptr<ISeCoRuleLearnerConfig> configPtr_;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const {
                return configPtr_->getCoverageStoppingCriterionConfig().get().createStoppingCriterionFactory();
            }

        public:

            /**
             * @param configPtr An unique pointer to an object of type `ISeCoRuleLearnerConfig`
             */
            SeCoRuleLearnerConfigurator(std::unique_ptr<ISeCoRuleLearnerConfig> configPtr)
                : RuleLearnerConfigurator(*configPtr), configPtr_(std::move(configPtr)) {}

            /**
             * @see `RuleLearnerConfigurator::createStoppingCriterionFactories`
             */
            void createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const override {
                RuleLearnerConfigurator::createStoppingCriterionFactories(factory);
                std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
                  this->createCoverageStoppingCriterionFactory();

                if (stoppingCriterionFactory) {
                    factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
                }
            }

            /**
             * @see `RuleLearnerConfigurator::createStatisticsProviderFactory`
             */
            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getHeadConfig().get().createStatisticsProviderFactory(labelMatrix);
            }

            /**
             * @see `RuleLearnerConfigurator::createModelBuilderFactory`
             */
            std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const override {
                return std::make_unique<DecisionListBuilderFactory>();
            }

            /**
             * @see `RuleLearnerConfigurator::createSparseBinaryPredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparseBinaryPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
                return configPtr_->getBinaryPredictorConfig().get().createSparsePredictorFactory(featureMatrix,
                                                                                                 numLabels);
            }
    };

    /**
     * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
     */
    class SeCoRuleLearnerConfig : public RuleLearnerConfig,
                                  virtual public ISeCoRuleLearnerConfig {
        protected:

            /**
             * An unique pointer that stores the configuration of the stopping criterion that stops the induction of
             * rules as soon as the sum of the weights of the uncovered labels is smaller or equal to a certain
             * threshold.
             */
            std::unique_ptr<IStoppingCriterionConfig> coverageStoppingCriterionConfigPtr_;

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
             * An unique pointer that stores the configuration of the lift function that affects the quality of rules,
             * depending on the number of labels for which they predict.
             */
            std::unique_ptr<ILiftFunctionConfig> liftFunctionConfigPtr_;

        public:

            SeCoRuleLearnerConfig()
                : RuleLearnerConfig(SECO_RULE_COMPARE_FUNCTION),
                  coverageStoppingCriterionConfigPtr_(std::make_unique<NoStoppingCriterionConfig>()),
                  headConfigPtr_(std::make_unique<SingleOutputHeadConfig>(
                    readableProperty(heuristicConfigPtr_), readableProperty(pruningHeuristicConfigPtr_))),
                  heuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                  pruningHeuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                  liftFunctionConfigPtr_(std::make_unique<NoLiftFunctionConfig>()) {}

            virtual ~SeCoRuleLearnerConfig() override {}

            Property<IStoppingCriterionConfig> getCoverageStoppingCriterionConfig() override final {
                return property(coverageStoppingCriterionConfigPtr_);
            }

            Property<IHeadConfig> getHeadConfig() override final {
                return property(headConfigPtr_);
            }

            Property<IHeuristicConfig> getHeuristicConfig() override final {
                return property(heuristicConfigPtr_);
            }

            Property<IHeuristicConfig> getPruningHeuristicConfig() override final {
                return property(pruningHeuristicConfigPtr_);
            }

            Property<ILiftFunctionConfig> getLiftFunctionConfig() override final {
                return property(liftFunctionConfigPtr_);
            }
    };
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
