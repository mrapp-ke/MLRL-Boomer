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
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/output/predictor_regression_label_wise.hpp"
#include "boosting/output/predictor_probability_auto.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/output/predictor_probability_marginalized.hpp"
#include "boosting/rule_evaluation/head_type_auto.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/rule_evaluation/regularization_no.hpp"
#include "boosting/rule_model_assemblage/default_rule_auto.hpp"
#include "boosting/statistics/statistic_format_auto.hpp"
#include "boosting/statistics/statistic_format_dense.hpp"
#include "boosting/statistics/statistic_format_sparse.hpp"
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/output/label_space_info_no.hpp"


namespace boosting {

    BoostingRuleLearner::Config::Config() {
        this->useAutomaticDefaultRule();
        this->useAutomaticFeatureBinning();
        this->useFeatureSamplingWithoutReplacement();
        this->useSizeStoppingCriterion();
        this->useConstantShrinkagePostProcessor();
        this->useAutomaticParallelRuleRefinement();
        this->useAutomaticParallelStatisticUpdate();
        this->useAutomaticHeads();
        this->useAutomaticStatistics();
        this->useNoL1Regularization();
        this->useL2Regularization();
        this->useLabelWiseLogisticLoss();
        this->useAutomaticLabelBinning();
        this->useAutomaticClassificationPredictor();
        this->useLabelWiseRegressionPredictor();
        this->useAutomaticProbabilityPredictor();
    }

    const IHeadConfig& BoostingRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IStatisticsConfig& BoostingRuleLearner::Config::getStatisticsConfig() const {
        return *statisticsConfigPtr_;
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

    IBeamSearchTopDownRuleInductionConfig& BoostingRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
            std::make_unique<BeamSearchTopDownRuleInductionConfig>(this->parallelRuleRefinementConfigPtr_);
        IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
        this->ruleInductionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualWidthFeatureBinningConfig& BoostingRuleLearner::Config::useEqualWidthFeatureBinning() {
        std::unique_ptr<EqualWidthFeatureBinningConfig> ptr =
            std::make_unique<EqualWidthFeatureBinningConfig>(this->parallelStatisticUpdateConfigPtr_);
        IEqualWidthFeatureBinningConfig& ref = *ptr;
        this->featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualFrequencyFeatureBinningConfig& BoostingRuleLearner::Config::useEqualFrequencyFeatureBinning() {
        std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
            std::make_unique<EqualFrequencyFeatureBinningConfig>(this->parallelStatisticUpdateConfigPtr_);
        IEqualFrequencyFeatureBinningConfig& ref = *ptr;
        this->featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelSamplingWithoutReplacementConfig& BoostingRuleLearner::Config::useLabelSamplingWithoutReplacement() {
        std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
            std::make_unique<LabelSamplingWithoutReplacementConfig>();
        ILabelSamplingWithoutReplacementConfig& ref = *ptr;
        this->labelSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithReplacementConfig& BoostingRuleLearner::Config::useInstanceSamplingWithReplacement() {
        std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithReplacementConfig>();
        IInstanceSamplingWithReplacementConfig& ref = *ptr;
        this->instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithoutReplacementConfig& BoostingRuleLearner::Config::useInstanceSamplingWithoutReplacement() {
        std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithoutReplacementConfig>();
        IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
        this->instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseStratifiedInstanceSamplingConfig& BoostingRuleLearner::Config::useLabelWiseStratifiedInstanceSampling() {
        std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
        ILabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        this->instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseStratifiedInstanceSamplingConfig& BoostingRuleLearner::Config::useExampleWiseStratifiedInstanceSampling() {
        std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
        IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        this->instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ISizeStoppingCriterionConfig& BoostingRuleLearner::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = AbstractRuleLearner::Config::useSizeStoppingCriterion();
        ref.setMaxRules(1000);
        return ref;
    }

    void BoostingRuleLearner::Config::useNoDefaultRule() {
        defaultRuleConfigPtr_ = std::make_unique<DefaultRuleConfig>(false);
    }

    void BoostingRuleLearner::Config::useAutomaticDefaultRule() {
        defaultRuleConfigPtr_ = std::make_unique<AutomaticDefaultRuleConfig>(statisticsConfigPtr_, lossConfigPtr_,
                                                                             headConfigPtr_);
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
        parallelRuleRefinementConfigPtr_ = std::make_unique<AutoParallelRuleRefinementConfig>(
            lossConfigPtr_, headConfigPtr_, featureSamplingConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticParallelStatisticUpdate() {
        parallelStatisticUpdateConfigPtr_ = std::make_unique<AutoParallelStatisticUpdateConfig>(lossConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticHeads() {
        headConfigPtr_ = std::make_unique<AutomaticHeadConfig>(
            lossConfigPtr_, labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void BoostingRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    IFixedPartialHeadConfig& BoostingRuleLearner::Config::useFixedPartialHeads() {
        std::unique_ptr<FixedPartialHeadConfig> ptr = std::make_unique<FixedPartialHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_);
        IFixedPartialHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    IDynamicPartialHeadConfig& BoostingRuleLearner::Config::useDynamicPartialHeads() {
        std::unique_ptr<DynamicPartialHeadConfig> ptr = std::make_unique<DynamicPartialHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_);
        IDynamicPartialHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    void BoostingRuleLearner::Config::useCompleteHeads() {
        headConfigPtr_ = std::make_unique<CompleteHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticStatistics() {
        statisticsConfigPtr_ =
            std::make_unique<AutomaticStatisticsConfig>(lossConfigPtr_, headConfigPtr_, defaultRuleConfigPtr_);
    }

    void BoostingRuleLearner::Config::useDenseStatistics() {
        statisticsConfigPtr_ = std::make_unique<DenseStatisticsConfig>(lossConfigPtr_);
    }

    void BoostingRuleLearner::Config::useSparseStatistics() {
        statisticsConfigPtr_ = std::make_unique<SparseStatisticsConfig>(lossConfigPtr_);
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

    void BoostingRuleLearner::Config::useExampleWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<ExampleWiseClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<AutomaticClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseRegressionPredictor() {
        regressionPredictorConfigPtr_ =
            std::make_unique<LabelWiseRegressionPredictorConfig>(parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<LabelWiseProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useMarginalizedProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<MarginalizedProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void BoostingRuleLearner::Config::useAutomaticProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<AutomaticProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    BoostingRuleLearner::BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr,
                                             Blas::DdotFunction ddotFunction, Blas::DspmvFunction dspmvFunction,
                                             Lapack::DsysvFunction dsysvFunction)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)), blas_(Blas(ddotFunction, dspmvFunction)),
          lapack_(Lapack(dsysvFunction)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> BoostingRuleLearner::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return configPtr_->getStatisticsConfig()
            .createStatisticsProviderFactory(featureMatrix, labelMatrix, blas_, lapack_);
    }

    std::unique_ptr<IModelBuilderFactory> BoostingRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<RuleListBuilderFactory>();
    }

    std::unique_ptr<IClassificationPredictorFactory> BoostingRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return configPtr_->getClassificationPredictorConfig()
            .createClassificationPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<IRegressionPredictorFactory> BoostingRuleLearner::createRegressionPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return configPtr_->getRegressionPredictorConfig().createRegressionPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<IProbabilityPredictorFactory> BoostingRuleLearner::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return configPtr_->getProbabilityPredictorConfig().createProbabilityPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<ILabelSpaceInfo> BoostingRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (configPtr_->getClassificationPredictorConfig().isLabelVectorSetNeeded()
            || configPtr_->getProbabilityPredictorConfig().isLabelVectorSetNeeded()
            || configPtr_->getRegressionPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig() {
        return std::make_unique<BoostingRuleLearner::Config>();
    }

    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr, Blas::DdotFunction ddotFunction,
            Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        return std::make_unique<BoostingRuleLearner>(std::move(configPtr), ddotFunction, dspmvFunction, dsysvFunction);
    }

}
