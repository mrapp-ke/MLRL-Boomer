/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/seco/learner.hpp"
#include "mlrl/seco/model/decision_list_builder.hpp"
#include "mlrl/seco/rule_evaluation/rule_compare_function.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * Allows to configure the individual modules of a rule learner that makes use of the separate-and-conquer (SeCo)
     * paradigm, depending on an `ISeCoRuleLearner::IConfig`.
     */
    class SeCoRuleLearnerConfigurator final : public RuleLearnerConfigurator {
        private:

            std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr_;

            std::unique_ptr<IStoppingCriterionFactory> createCoverageStoppingCriterionFactory() const {
                std::unique_ptr<CoverageStoppingCriterionConfig>& configPtr =
                  configPtr_->getCoverageStoppingCriterionConfigPtr();
                return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
            }

        public:

            /**
             * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig`
             */
            SeCoRuleLearnerConfigurator(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
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
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override {
                return configPtr_->getHeadConfigPtr()->createStatisticsProviderFactory(labelMatrix);
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
                return configPtr_->getBinaryPredictorConfigPtr()->createSparsePredictorFactory(featureMatrix,
                                                                                               numLabels);
            }
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

                public:

                    Config()
                        : AbstractRuleLearner::Config(SECO_RULE_COMPARE_FUNCTION),
                          headConfigPtr_(
                            std::make_unique<SingleOutputHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_)),
                          heuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                          pruningHeuristicConfigPtr_(std::make_unique<PrecisionConfig>()),
                          liftFunctionConfigPtr_(std::make_unique<NoLiftFunctionConfig>()) {}

                    virtual ~Config() override {}

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

        public:

            /**
             * @param configurator A reference to an object of type `SeCoRuleLearnerConfigurator` that allows to
             *                     configure the individual modules to be used by the rule learner
             */
            explicit AbstractSeCoRuleLearner(SeCoRuleLearnerConfigurator& configurator)
                : AbstractRuleLearner(configurator) {}

            virtual ~AbstractSeCoRuleLearner() override {}
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
