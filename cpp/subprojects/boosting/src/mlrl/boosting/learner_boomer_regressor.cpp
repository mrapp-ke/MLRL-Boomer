#include "mlrl/boosting/learner_boomer_regressor.hpp"

#include "mlrl/boosting/learner_common.hpp"
#include "mlrl/common/learner_regression_common.hpp"

namespace boosting {

    /**
     * The BOOMER algorithm for regression problems.
     */
    class BoomerRegressor final : public AbstractRegressionRuleLearner,
                                  virtual public IBoomerRegressor {
        public:

            /**
             * Allows to configure the BOOMER algorithm for classification problems.
             */
            class Config final : public BoostedRuleLearnerConfig,
                                 virtual public IBoomerRegressor::IConfig {
                public:

                    void useDefaults() override {
                        IBoostedRuleLearnerMixin::useDefaults();
                        this->useAutomaticDefaultRule();
                        this->useSequentialRuleModelAssemblage();
                        this->useGreedyTopDownRuleInduction();
                        this->useAutomaticFeatureBinning();
                        this->useFeatureSamplingWithoutReplacement();
                        this->useAutomaticPartitionSampling();
                        this->useConstantShrinkagePostProcessor();
                        this->useAutomaticParallelRuleRefinement();
                        this->useAutomaticParallelStatisticUpdate();
                        this->useParallelPrediction();
                        this->useSizeStoppingCriterion();
                        this->useOutputWiseScorePredictor();
                        this->useAutomaticHeads();
                        this->useAutomaticStatistics();
                        this->useDecomposableSquaredErrorLoss();
                        this->useL2Regularization();
                    }

                    /**
                     * @see `ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override {
                        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
                        ref.setMaxRules(1000);
                        return ref;
                    }
            };

        private:

            const std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr_;

        public:

            /**
             * @param configuratorPtr An unique pointer to an object of type `BoostedRuleLearnerConfigurator` that
             *                        allows to configure the individual modules to be used by the rule learner
             */
            BoomerRegressor(std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr)
                : AbstractRegressionRuleLearner(*configuratorPtr), configuratorPtr_(std::move(configuratorPtr)) {}
    };

    std::unique_ptr<IBoomerRegressor::IConfig> createBoomerRegressorConfig() {
        auto ptr = std::make_unique<BoomerRegressor::Config>();
        ptr->useDefaults();
        return ptr;
    }

    std::unique_ptr<IBoomerRegressor> createBoomerRegressor(std::unique_ptr<IBoomerRegressor::IConfig> configPtr,
                                                            Blas::DdotFunction ddotFunction,
                                                            Blas::DspmvFunction dspmvFunction,
                                                            Lapack::DsysvFunction dsysvFunction) {
        std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr =
          std::make_unique<BoostedRuleLearnerConfigurator>(std::move(configPtr), ddotFunction, dspmvFunction,
                                                           dsysvFunction);
        return std::make_unique<BoomerRegressor>(std::move(configuratorPtr));
    }

}
