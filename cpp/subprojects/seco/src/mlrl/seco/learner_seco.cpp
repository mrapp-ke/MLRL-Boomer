#include "mlrl/seco/learner_seco.hpp"

#include "mlrl/seco/learner_common.hpp"

namespace seco {

    /**
     * The multi-label SeCo algorithm.
     */
    class MultiLabelSeCoRuleLearner final : public AbstractSeCoRuleLearner,
                                            virtual public IMultiLabelSeCoRuleLearner {
        public:

            /**
             * Allows to configure the multi-label SeCo algorithm.
             */
            class Config final : public AbstractSeCoRuleLearner::Config,
                                 virtual public IMultiLabelSeCoRuleLearner::IConfig {
                public:

                    Config() {
                        this->useSequentialRuleModelAssemblage();
                        this->useGreedyTopDownRuleInduction();
                        this->useDefaultRule();
                        this->useNoOutputSampling();
                        this->useNoInstanceSampling();
                        this->useNoFeatureSampling();
                        this->useNoPartitionSampling();
                        this->useGreedyTopDownRuleInduction();
                        this->useCoverageStoppingCriterion();
                        this->useSizeStoppingCriterion();
                        this->useNoTimeStoppingCriterion();
                        this->useOutputWiseStratifiedInstanceSampling();
                        this->useSingleOutputHeads();
                        this->useIrepRulePruning();
                        this->useNoSequentialPostOptimization();
                        this->useFMeasureHeuristic();
                        this->useAccuracyPruningHeuristic();
                        this->usePeakLiftFunction();
                        this->useParallelRuleRefinement();
                        this->useNoParallelStatisticUpdate();
                        this->useParallelPrediction();
                        this->useOutputWiseBinaryPredictor();
                    }

                    /**
                     * @see `IRuleLearner::IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override {
                        IGreedyTopDownRuleInductionConfig& ref =
                          IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction();
                        ref.setRecalculatePredictions(false);
                        return ref;
                    }

                    /**
                     * @see `IRuleLearner::IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction`
                     */
                    IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() override {
                        IBeamSearchTopDownRuleInductionConfig& ref =
                          IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction();
                        ref.setRecalculatePredictions(false);
                        return ref;
                    }

                    /**
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override {
                        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
                        ref.setMaxRules(500);
                        return ref;
                    }
            };

        private:

            const std::unique_ptr<SeCoRuleLearnerConfigurator> configuratorPtr_;

        public:

            /**
             * @param configuratorPtr An unique pointer to an object of type `SeCoRuleLearnerConfigurator` that allows
             *                        to configure the individual modules to be used by the rule learner
             */
            MultiLabelSeCoRuleLearner(std::unique_ptr<SeCoRuleLearnerConfigurator> configuratorPtr)
                : AbstractSeCoRuleLearner(*configuratorPtr), configuratorPtr_(std::move(configuratorPtr)) {}
    };

    std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> createMultiLabelSeCoRuleLearnerConfig() {
        return std::make_unique<MultiLabelSeCoRuleLearner::Config>();
    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner> createMultiLabelSeCoRuleLearner(
      std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr) {
        std::unique_ptr<SeCoRuleLearnerConfigurator> configuratorPtr =
          std::make_unique<SeCoRuleLearnerConfigurator>(std::move(configPtr));
        return std::make_unique<MultiLabelSeCoRuleLearner>(std::move(configuratorPtr));
    }

}
