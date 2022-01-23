#include "boosting/learner.hpp"
#include "boosting/binning/feature_binning_auto.hpp"
#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/binning/label_binning_no.hpp"
#include "boosting/losses/loss_example_wise_logistic.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/losses/loss_label_wise_squared_error.hpp"
#include "boosting/losses/loss_label_wise_squared_hinge.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/rule_evaluation/regularization_no.hpp"


namespace boosting {

    BoostingRuleLearner::Config::Config() {
        this->useAutomaticFeatureBinning();
        this->useFeatureSamplingWithoutReplacement();
        this->useSizeStoppingCriterion();
        this->useConstantShrinkagePostProcessor();
        this->useSingleLabelHeads();
        this->useNoL1Regularization();
        this->useL2Regularization();
        this->useLabelWiseLogisticLoss();
        this->useAutomaticLabelBinning();
        this->useAutomaticClassificationPredictor();
        this->useLabelWiseRegressionPredictor();
        this->useLabelWiseProbabilityPredictor();
    }

    const IHeadConfig& BoostingRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IRegularizationConfig& BoostingRuleLearner::Config::getL1RegularizationConfig() const {
        return *l1RegularizationConfigPtr_;
    }

    const IRegularizationConfig& BoostingRuleLearner::Config::getL2RegularizationConfig() const {
        return *l2RegularizationConfigPtr_;
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
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void BoostingRuleLearner::Config::useCompleteHeads() {
        headConfigPtr_ = std::make_unique<CompleteHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void BoostingRuleLearner::Config::useNoL1Regularization() {
        l1RegularizationConfigPtr_ = std::make_unique<NoRegularizationConfig>();
    }

    IManualRegularizationConfig& BoostingRuleLearner::Config::useL1Regularization() {
        std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
        IManualRegularizationConfig& ref = *ptr;
        l1RegularizationConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useNoL2Regularization() {
        l2RegularizationConfigPtr_ = std::make_unique<NoRegularizationConfig>();
    }

    IManualRegularizationConfig& BoostingRuleLearner::Config::useL2Regularization() {
        std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
        IManualRegularizationConfig& ref = *ptr;
        l2RegularizationConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useExampleWiseLogisticLoss() {
        lossConfigPtr_ = std::make_unique<ExampleWiseLogisticLossConfig>(headConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseLogisticLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseLogisticLossConfig>(headConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseSquaredErrorLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseSquaredErrorLossConfig>(headConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseSquaredHingeLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseSquaredHingeLossConfig>(headConfigPtr_);
    }

    void BoostingRuleLearner::Config::useNoLabelBinning() {
        labelBinningConfigPtr_ =
            std::make_unique<NoLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticLabelBinning() {
        labelBinningConfigPtr_ =
            std::make_unique<AutomaticLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    IEqualWidthLabelBinningConfig& BoostingRuleLearner::Config::useEqualWidthLabelBinning() {
        std::unique_ptr<EqualWidthLabelBinningConfig> ptr =
            std::make_unique<EqualWidthLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
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

    void BoostingRuleLearner::Config::useAutomaticClassificationPredictor() {
        classificationPredictorConfigPtr_ = std::make_unique<AutomaticClassificationPredictorConfig>();
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
        return configPtr_->getLossConfig().configure(labelMatrix);
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
