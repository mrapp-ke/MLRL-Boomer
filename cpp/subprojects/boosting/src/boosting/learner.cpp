#include "boosting/learner.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"
#include "common/post_processing/post_processor_no.hpp"


namespace boosting {

    BoostingRuleLearner::Config::Config() {
        this->useSizeStoppingCriterion();
        this->useFeatureSamplingWithoutReplacement();
        this->useConstantShrinkagePostProcessor();
        this->useLabelWiseLogisticLoss();
        this->useNoLabelBinning();
        this->useLabelWiseClassificationPredictor(); // TODO use automatical configuration by default
        this->useLabelWiseRegressionPredictor();
        this->useLabelWiseProbabilityPredictor();
    }

    const IPostProcessorConfig& BoostingRuleLearner::Config::getPostProcessorConfig() const {
        return *postProcessorConfigPtr_;
    }

    const ILossConfig& BoostingRuleLearner::Config::getLossConfig() const {
        return *lossConfigPtr_;
    }

    const ILabelBinningConfig* BoostingRuleLearner::Config::getLabelBinningConfig() const {
        return labelBinningConfigPtr_.get();
    }

    const IClassificationPredictorConfig& BoostingRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    const IRegressionPredictorConfig& BoostingRuleLearner::Config::getRegressionPredictorConfig() const {
        return *regressionPredictorConfigPtr_;
    }

    const IProbabilityPredictorConfig& BoostingRuleLearner::Config::getProbabilityPredictorConfig() const {
        return *probabilityPredictorConfigPtr_;
    }

    ISizeStoppingCriterionConfig& BoostingRuleLearner::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = AbstractRuleLearner::Config::useSizeStoppingCriterion();
        ref.setMaxRules(1000);
        return ref;
    }

    void BoostingRuleLearner::Config::useNoPostProcessor() {
        postProcessorConfigPtr_ = std::make_unique<NoPostProcessorConfig>();
    }

    ConstantShrinkageConfig& BoostingRuleLearner::Config::useConstantShrinkagePostProcessor() {
        std::unique_ptr<ConstantShrinkageConfig> ptr = std::make_unique<ConstantShrinkageConfig>();
        ConstantShrinkageConfig& ref = *ptr;
        postProcessorConfigPtr_ = std::move(ptr);
        return ref;
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

    ExampleWiseClassificationPredictorConfig& BoostingRuleLearner::Config::useExampleWiseClassificationPredictor() {
        std::unique_ptr<ExampleWiseClassificationPredictorConfig> ptr
            = std::make_unique<ExampleWiseClassificationPredictorConfig>();
        ExampleWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseClassificationPredictorConfig& BoostingRuleLearner::Config::useLabelWiseClassificationPredictor() {
        std::unique_ptr<LabelWiseClassificationPredictorConfig> ptr =
            std::make_unique<LabelWiseClassificationPredictorConfig>();
        LabelWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseRegressionPredictorConfig& BoostingRuleLearner::Config::useLabelWiseRegressionPredictor() {
        std::unique_ptr<LabelWiseRegressionPredictorConfig> ptr
            = std::make_unique<LabelWiseRegressionPredictorConfig>();
        LabelWiseRegressionPredictorConfig& ref = *ptr;
        regressionPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseProbabilityPredictorConfig& BoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        std::unique_ptr<LabelWiseProbabilityPredictorConfig> ptr
            = std::make_unique<LabelWiseProbabilityPredictorConfig>();
        LabelWiseProbabilityPredictorConfig& ref = *ptr;
        probabilityPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    BoostingRuleLearner::BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IPostProcessorFactory> BoostingRuleLearner::createPostProcessorFactory() const {
        const IPostProcessorConfig* baseConfig = &configPtr_->getPostProcessorConfig();

        if (dynamic_cast<const NoPostProcessorConfig*>(baseConfig)) {
            return std::make_unique<NoPostProcessorFactory>();
        } else if (auto* config = dynamic_cast<const ConstantShrinkageConfig*>(baseConfig)) {
            return std::make_unique<ConstantShrinkageFactory>(config->getShrinkage());
        }

        throw std::runtime_error("Failed to create IPostProcessorFactory");
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
        const IClassificationPredictorConfig* baseConfig = &configPtr_->getClassificationPredictorConfig();

        if (auto* config = dynamic_cast<const LabelWiseClassificationPredictorConfig*>(baseConfig)) {
            float64 threshold = 0;  // TODO Use correct threshold
            return std::make_unique<LabelWiseClassificationPredictorFactory>(threshold, config->getNumThreads());
        }

        throw std::runtime_error("Failed to create IClassificationPredictorFactory");
    }

    std::unique_ptr<IRegressionPredictorFactory> BoostingRuleLearner::createRegressionPredictorFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IProbabilityPredictorFactory> BoostingRuleLearner::createProbabilityPredictorFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig() {
        return std::make_unique<BoostingRuleLearner::Config>();
    }

    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr) {
        return std::make_unique<BoostingRuleLearner>(std::move(configPtr));
    }

}
