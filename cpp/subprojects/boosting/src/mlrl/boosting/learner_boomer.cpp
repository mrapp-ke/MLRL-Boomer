#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/boosting/learner_boomer.hpp"

#include "mlrl/boosting/learner_common.hpp"
#include "mlrl/common/learner_classification_common.hpp"

namespace boosting {

    /**
     * The BOOMER algorithm.
     */
    class Boomer final : public AbstractClassificationRuleLearner,
                         virtual public IBoomer {
        public:

            /**
             * Allows to configure the BOOMER algorithm.
             */
            class Config final : public BoostedRuleLearnerConfig,
                                 virtual public IBoomer::IConfig {
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
                        this->useDecomposableLogisticLoss();
                        this->useNoL1Regularization();
                        this->useL2Regularization();
                        this->useAutomaticLabelBinning();
                        this->useAutomaticBinaryPredictor();
                        this->useOutputWiseScorePredictor();
                        this->useAutomaticProbabilityPredictor();
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
            Boomer(std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr)
                : AbstractClassificationRuleLearner(*configuratorPtr), configuratorPtr_(std::move(configuratorPtr)) {}
    };

    std::unique_ptr<IBoomer::IConfig> createBoomerConfig() {
        return std::make_unique<Boomer::Config>();
    }

    std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                                          Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr =
          std::make_unique<BoostedRuleLearnerConfigurator>(std::move(configPtr), ddotFunction, dspmvFunction,
                                                           dsysvFunction);
        return std::make_unique<Boomer>(std::move(configuratorPtr));
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
