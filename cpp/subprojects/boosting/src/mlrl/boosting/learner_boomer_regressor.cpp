#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

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

                    Config() {
                        this->useSequentialRuleModelAssemblage();
                        this->useGreedyTopDownRuleInduction();
                        this->useDefaultRule();
                        this->useNoOutputSampling();
                        this->useNoInstanceSampling();
                        this->useFeatureSamplingWithoutReplacement();
                        this->useParallelPrediction();
                        this->useAutomaticDefaultRule();
                        this->useAutomaticPartitionSampling();
                        this->useAutomaticFeatureBinning();
                        this->useSizeStoppingCriterion();
                        this->useNoTimeStoppingCriterion();
                        this->useNoRulePruning();
                        this->useNoGlobalPruning();
                        this->useNoSequentialPostOptimization();
                        this->useConstantShrinkagePostProcessor();
                        this->useAutomaticParallelRuleRefinement();
                        this->useAutomaticParallelStatisticUpdate();
                        this->useAutomaticHeads();
                        this->useAutomaticStatistics();
                        this->useDecomposableSquaredErrorLoss();
                        this->useNoL1Regularization();
                        this->useL2Regularization();
                        this->useOutputWiseScorePredictor();
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
        return std::make_unique<BoomerRegressor::Config>();
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

#ifdef _WIN32
    #pragma warning(pop)
#endif
