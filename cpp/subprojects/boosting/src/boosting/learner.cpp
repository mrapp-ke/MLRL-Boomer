#include "boosting/learner.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"


namespace boosting {

    BoostingRuleLearner::Config::Config() {
        this->useLabelWiseLogisticLoss();
        this->useNoLabelBinning();
    }

    const ILabelBinningConfig* BoostingRuleLearner::Config::getLabelBinningConfig() const {
        return labelBinningConfigPtr_.get();
    }

    const ILossConfig& BoostingRuleLearner::Config::getLossConfig() const {
        return *lossConfigPtr_;
    }

    ExampleWiseLogisticLossConfig& BoostingRuleLearner::Config::useExampleWiseLogisticLoss() {
        std::unique_ptr<ExampleWiseLogisticLossConfig> ptr = std::make_unique<ExampleWiseLogisticLossConfig>();
        ExampleWiseLogisticLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseLogisticLossConfig& BoostingRuleLearner::Config::useLabelWiseLogisticLoss() {
        std::unique_ptr<LabelWiseLogisticLossConfig> ptr = std::make_unique<LabelWiseLogisticLossConfig>();
        LabelWiseLogisticLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseSquaredErrorLossConfig& BoostingRuleLearner::Config::useLabelWiseSquaredErrorLoss() {
        std::unique_ptr<LabelWiseSquaredErrorLossConfig> ptr = std::make_unique<LabelWiseSquaredErrorLossConfig>();
        LabelWiseSquaredErrorLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseSquaredHingeLossConfig& BoostingRuleLearner::Config::useLabelWiseSquaredHingeLoss() {
        std::unique_ptr<LabelWiseSquaredHingeLossConfig> ptr = std::make_unique<LabelWiseSquaredHingeLossConfig>();
        LabelWiseSquaredHingeLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useNoLabelBinning() {
        labelBinningConfigPtr_ = nullptr;
    }

    EqualWidthLabelBinningConfig& BoostingRuleLearner::Config::useEqualWidthLabelBinning() {
        std::unique_ptr<EqualWidthLabelBinningConfig> ptr = std::make_unique<EqualWidthLabelBinningConfig>();
        EqualWidthLabelBinningConfig& ref = *ptr;
        labelBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    std::unique_ptr<IStatisticsProviderFactory> BoostingRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        float64 l1RegularizationWeight = 0;
        float64 l2RegularizationWeight = 1;
        uint32 numThreads = 1;
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
            std::make_unique<LabelWiseLogisticLossFactory>(),
            std::make_unique<LabelWiseLogisticLossFactory>(),
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

    BoostingRuleLearner::BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(std::move(configPtr)) {

    }

    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig() {
        return std::make_unique<BoostingRuleLearner::Config>();
    }

    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr) {
        return std::make_unique<BoostingRuleLearner>(std::move(configPtr));
    }

}
