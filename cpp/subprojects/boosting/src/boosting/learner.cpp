#include "boosting/learner.hpp"
#include "boosting/binning/feature_binning_auto.hpp"
#include "boosting/binning/label_binning_no.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"


namespace boosting {

    BoostingRuleLearner::Config::Config() {
        this->useAutomaticFeatureBinning();
        this->useFeatureSamplingWithoutReplacement();
        this->useSizeStoppingCriterion();
        this->useConstantShrinkagePostProcessor();
        this->useSingleLabelHeads();
        this->useLabelWiseLogisticLoss();
        this->useNoLabelBinning(); // TODO use automatic configuration by default
        this->useLabelWiseClassificationPredictor(); // TODO use automatic configuration by default
        this->useLabelWiseRegressionPredictor();
        this->useLabelWiseProbabilityPredictor();
    }

    const IHeadConfig& BoostingRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const ILossConfig& BoostingRuleLearner::Config::getLossConfig() const {
        return *lossConfigPtr_;
    }

    const ILabelBinningConfig& BoostingRuleLearner::Config::getLabelBinningConfig() const {
        return *labelBinningConfigPtr_;
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

    void BoostingRuleLearner::Config::useAutomaticFeatureBinning() {
        featureBinningConfigPtr_ = std::make_unique<AutomaticFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
    }

    IConstantShrinkageConfig& BoostingRuleLearner::Config::useConstantShrinkagePostProcessor() {
        std::unique_ptr<ConstantShrinkageConfig> ptr = std::make_unique<ConstantShrinkageConfig>();
        IConstantShrinkageConfig& ref = *ptr;
        postProcessorConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useAutomaticParallelRuleRefinement() {
        parallelRuleRefinementConfigPtr_ = std::make_unique<AutoParallelRuleRefinementConfig>();
    }

    void BoostingRuleLearner::Config::useAutomaticParallelStatisticUpdate() {
        parallelStatisticUpdateConfigPtr_ = std::make_unique<AutoParallelStatisticUpdateConfig>();
    }

    void BoostingRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(labelBinningConfigPtr_,
                                                                 parallelStatisticUpdateConfigPtr_);
    }

    void BoostingRuleLearner::Config::useCompleteHeads() {
        headConfigPtr_ = std::make_unique<CompleteHeadConfig>(labelBinningConfigPtr_,
                                                              parallelStatisticUpdateConfigPtr_);
    }

    IExampleWiseLogisticLossConfig& BoostingRuleLearner::Config::useExampleWiseLogisticLoss() {
        std::unique_ptr<ExampleWiseLogisticLossConfig> ptr =
            std::make_unique<ExampleWiseLogisticLossConfig>(headConfigPtr_);
        IExampleWiseLogisticLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseLogisticLossConfig& BoostingRuleLearner::Config::useLabelWiseLogisticLoss() {
        std::unique_ptr<LabelWiseLogisticLossConfig> ptr =
            std::make_unique<LabelWiseLogisticLossConfig>(headConfigPtr_);
        ILabelWiseLogisticLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseSquaredErrorLossConfig& BoostingRuleLearner::Config::useLabelWiseSquaredErrorLoss() {
        std::unique_ptr<LabelWiseSquaredErrorLossConfig> ptr =
            std::make_unique<LabelWiseSquaredErrorLossConfig>(headConfigPtr_);
        ILabelWiseSquaredErrorLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseSquaredHingeLossConfig& BoostingRuleLearner::Config::useLabelWiseSquaredHingeLoss() {
        std::unique_ptr<LabelWiseSquaredHingeLossConfig> ptr =
            std::make_unique<LabelWiseSquaredHingeLossConfig>(headConfigPtr_);
        ILabelWiseSquaredHingeLossConfig& ref = *ptr;
        lossConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useNoLabelBinning() {
        labelBinningConfigPtr_ = std::make_unique<NoLabelBinningConfig>();
    }

    IEqualWidthLabelBinningConfig& BoostingRuleLearner::Config::useEqualWidthLabelBinning() {
        std::unique_ptr<EqualWidthLabelBinningConfig> ptr = std::make_unique<EqualWidthLabelBinningConfig>();
        IEqualWidthLabelBinningConfig& ref = *ptr;
        labelBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseClassificationPredictorConfig& BoostingRuleLearner::Config::useExampleWiseClassificationPredictor() {
        std::unique_ptr<ExampleWiseClassificationPredictorConfig> ptr
            = std::make_unique<ExampleWiseClassificationPredictorConfig>();
        IExampleWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseClassificationPredictorConfig& BoostingRuleLearner::Config::useLabelWiseClassificationPredictor() {
        std::unique_ptr<LabelWiseClassificationPredictorConfig> ptr =
            std::make_unique<LabelWiseClassificationPredictorConfig>();
        ILabelWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseRegressionPredictorConfig& BoostingRuleLearner::Config::useLabelWiseRegressionPredictor() {
        std::unique_ptr<LabelWiseRegressionPredictorConfig> ptr
            = std::make_unique<LabelWiseRegressionPredictorConfig>();
        ILabelWiseRegressionPredictorConfig& ref = *ptr;
        regressionPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseProbabilityPredictorConfig& BoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        std::unique_ptr<LabelWiseProbabilityPredictorConfig> ptr
            = std::make_unique<LabelWiseProbabilityPredictorConfig>();
        ILabelWiseProbabilityPredictorConfig& ref = *ptr;
        probabilityPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    BoostingRuleLearner::BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> BoostingRuleLearner::createStatisticsProviderFactory(
            const IRowWiseLabelMatrix& labelMatrix) const {
        return configPtr_->getLossConfig().configure();
    }

    std::unique_ptr<IModelBuilder> BoostingRuleLearner::createModelBuilder() const {
        return std::make_unique<RuleListBuilder>();
    }

    std::unique_ptr<IClassificationPredictorFactory> BoostingRuleLearner::createClassificationPredictorFactory() const {
        return configPtr_->getClassificationPredictorConfig().configure();
    }

    std::unique_ptr<IRegressionPredictorFactory> BoostingRuleLearner::createRegressionPredictorFactory() const {
        return configPtr_->getRegressionPredictorConfig().configure();
    }

    std::unique_ptr<IProbabilityPredictorFactory> BoostingRuleLearner::createProbabilityPredictorFactory() const {
        return configPtr_->getProbabilityPredictorConfig().configure();
    }

    std::unique_ptr<ILabelSpaceInfo> BoostingRuleLearner::createLabelSpaceInfo() const {
        // TODO Implement
        return AbstractRuleLearner::createLabelSpaceInfo();
    }

    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig() {
        return std::make_unique<BoostingRuleLearner::Config>();
    }

    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr) {
        return std::make_unique<BoostingRuleLearner>(std::move(configPtr));
    }

}
