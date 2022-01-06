#include "boosting/learner.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_factory_label_wise_dense.hpp"


namespace boosting {

    BoostingRuleLearner::BoostingRuleLearner(std::unique_ptr<Config> configPtr)
        : AbstractRuleLearner(std::move(configPtr)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> BoostingRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        float64 l1RegularizationWeight = 0;
        float64 l2RegularizationWeight = 1;
        uint32 numThreads = 1;
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
            std::make_unique<LabelWiseLogisticLossFactory>(), std::make_unique<LabelWiseLogisticLossFactory>(),
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight),
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight),
            std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight),
            numThreads);
    }

    std::unique_ptr<IModelBuilder> BoostingRuleLearner::createModelBuilder() const {
        return std::make_unique<RuleListBuilder>();
    }

    std::unique_ptr<IClassificationPredictorFactory> BoostingRuleLearner::createClassificationPredictorFactory() const {
        // TODO Implement
        float64 threshold = 0;
        uint32 numThreads = 1;
        return std::make_unique<LabelWiseClassificationPredictorFactory>(threshold, numThreads);
    }

}
