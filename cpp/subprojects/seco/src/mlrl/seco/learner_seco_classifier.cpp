#include "mlrl/seco/learner_seco_classifier.hpp"

#include "mlrl/common/learner_classification_common.hpp"
#include "mlrl/seco/learner_common.hpp"

namespace seco {

    /**
     * The multi-label SeCo algorithm.
     */
    class SeCoClassifier final : public AbstractClassificationRuleLearner,
                                 virtual public ISeCoClassifier {
        public:

            /**
             * Allows to configure the multi-label SeCo algorithm.
             */
            class Config final : public SeCoRuleLearnerConfig,
                                 virtual public ISeCoClassifier::IConfig {
                public:

                    void useDefaults() override {
                        ISeCoRuleLearnerMixin::useDefaults();
                        this->useSequentialRuleModelAssemblage();
                        this->useGreedyTopDownRuleInduction();
                        this->useOutputWiseStratifiedInstanceSampling();
                        this->useIrepRulePruning();
                        this->useParallelRuleRefinement();
                        this->useParallelPrediction();
                        this->useSizeStoppingCriterion();
                        this->useOutputWiseBinaryPredictor();
                        this->useCoverageStoppingCriterion();
                        this->useSingleOutputHeads();
                        this->useFMeasureHeuristic();
                        this->useAccuracyPruningHeuristic();
                        this->usePeakLiftFunction();
                    }

                    /**
                     * @see `IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction`
                     */
                    IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() override {
                        IGreedyTopDownRuleInductionConfig& ref =
                          IGreedyTopDownRuleInductionMixin::useGreedyTopDownRuleInduction();
                        ref.setRecalculatePredictions(false);
                        return ref;
                    }

                    /**
                     * @see `IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction`
                     */
                    IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() override {
                        IBeamSearchTopDownRuleInductionConfig& ref =
                          IBeamSearchTopDownRuleInductionMixin::useBeamSearchTopDownRuleInduction();
                        ref.setRecalculatePredictions(false);
                        return ref;
                    }

                    /**
                     * @see `ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
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
            SeCoClassifier(std::unique_ptr<SeCoRuleLearnerConfigurator> configuratorPtr)
                : AbstractClassificationRuleLearner(*configuratorPtr), configuratorPtr_(std::move(configuratorPtr)) {}
    };

    std::unique_ptr<ISeCoClassifier::IConfig> createSeCoClassifierConfig() {
        auto ptr = std::make_unique<SeCoClassifier::Config>();
        ptr->useDefaults();
        return ptr;
    }

    std::unique_ptr<ISeCoClassifier> createSeCoClassifier(std::unique_ptr<ISeCoClassifier::IConfig> configPtr) {
        std::unique_ptr<SeCoRuleLearnerConfigurator> configuratorPtr =
          std::make_unique<SeCoRuleLearnerConfigurator>(std::move(configPtr));
        return std::make_unique<SeCoClassifier>(std::move(configuratorPtr));
    }

}
