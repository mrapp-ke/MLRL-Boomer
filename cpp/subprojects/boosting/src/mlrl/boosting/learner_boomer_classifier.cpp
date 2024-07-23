#include "mlrl/boosting/learner_boomer_classifier.hpp"

#include "mlrl/boosting/learner_common.hpp"
#include "mlrl/common/learner_classification_common.hpp"

namespace boosting {

    /**
     * The BOOMER algorithm for classification problems.
     */
    class BoomerClassifier final : public AbstractClassificationRuleLearner,
                                   virtual public IBoomerClassifier {
        public:

            /**
             * Allows to configure the BOOMER algorithm for classification problems.
             */
            class Config final : public BoostedRuleLearnerConfig,
                                 virtual public IBoomerClassifier::IConfig {
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
                        this->useAutomaticProbabilityPredictor();
                        this->useAutomaticBinaryPredictor();
                        this->useAutomaticHeads();
                        this->useAutomaticStatistics();
                        this->useDecomposableLogisticLoss();
                        this->useL2Regularization();
                        this->useAutomaticLabelBinning();
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
            BoomerClassifier(std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr)
                : AbstractClassificationRuleLearner(*configuratorPtr), configuratorPtr_(std::move(configuratorPtr)) {}
    };

    std::unique_ptr<IBoomerClassifier::IConfig> createBoomerClassifierConfig() {
        auto ptr = std::make_unique<BoomerClassifier::Config>();
        ptr->useDefaults();
        return ptr;
    }

    std::unique_ptr<IBoomerClassifier> createBoomerClassifier(std::unique_ptr<IBoomerClassifier::IConfig> configPtr,
                                                              Blas::DdotFunction ddotFunction,
                                                              Blas::DspmvFunction dspmvFunction,
                                                              Lapack::DsysvFunction dsysvFunction) {
        std::unique_ptr<BoostedRuleLearnerConfigurator> configuratorPtr =
          std::make_unique<BoostedRuleLearnerConfigurator>(std::move(configPtr), ddotFunction, dspmvFunction,
                                                           dsysvFunction);
        return std::make_unique<BoomerClassifier>(std::move(configuratorPtr));
    }

}
