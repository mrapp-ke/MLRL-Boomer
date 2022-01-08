#include "boosting/learner.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"


namespace boosting {

    /**
     * A rule learner that makes use of gradient boosting.
     */
    class BoostingRuleLearner final : public AbstractRuleLearner, virtual public IBoostingRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config, virtual public IBoostingRuleLearner::IConfig {

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override {
                // TODO Implement
                float64 l1RegularizationWeight = 0;
                float64 l2RegularizationWeight = 1;
                uint32 numThreads = 1;
                return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
                    std::make_unique<LabelWiseLogisticLossFactory>(),
                    std::make_unique<LabelWiseLogisticLossFactory>(),
                    std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                l2RegularizationWeight),
                    std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                l2RegularizationWeight),
                    std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight,
                                                                                l2RegularizationWeight),
                    numThreads);
            }

            std::unique_ptr<IModelBuilder> createModelBuilder() const override {
                return std::make_unique<RuleListBuilder>();
            }

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override {
                // TODO Implement
                float64 threshold = 0;
                uint32 numThreads = 1;
                return std::make_unique<LabelWiseClassificationPredictorFactory>(threshold, numThreads);
            }

        public:

            /**
             * @param configPtr An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that specifies
             *                  the configuration that should be used by the rule learner
             */
            BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr)
                : AbstractRuleLearner(std::move(configPtr)) {

            }

    };

    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig() {
        return std::make_unique<BoostingRuleLearner::Config>();
    }

    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr) {
        return std::make_unique<BoostingRuleLearner>(std::move(configPtr));
    }

}
