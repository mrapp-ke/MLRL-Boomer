#include "mlrl/boosting/learner_boomer.hpp"

#include "mlrl/boosting/learner_common.hpp"

namespace boosting {

    /**
     * The BOOMER algorithm.
     */
    class Boomer final : public AbstractBoostedRuleLearner,
                         virtual public IBoomer {
        public:

            /**
             * Allows to configure the BOOMER algorithm.
             */
            class Config final : public AbstractBoostedRuleLearner::Config,
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
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override {
                        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
                        ref.setMaxRules(1000);
                        return ref;
                    }
            };

        private:

            const std::unique_ptr<IBoomer::IConfig> configPtr_;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoomer::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction)
                : AbstractBoostedRuleLearner(*configPtr, ddotFunction, dspmvFunction, dsysvFunction),
                  configPtr_(std::move(configPtr)) {}
    };

    std::unique_ptr<IBoomer::IConfig> createBoomerConfig() {
        return std::make_unique<Boomer::Config>();
    }

    std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                                          Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        return std::make_unique<Boomer>(std::move(configPtr), ddotFunction, dspmvFunction, dsysvFunction);
    }

}
