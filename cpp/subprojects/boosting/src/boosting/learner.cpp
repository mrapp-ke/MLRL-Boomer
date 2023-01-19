#include "boosting/learner.hpp"
#include "boosting/binning/label_binning_no.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/output/predictor_regression_label_wise.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/regularization_no.hpp"
#include "boosting/rule_evaluation/rule_compare_function.hpp"
#include "boosting/statistics/statistic_format_dense.hpp"
#include "common/prediction/label_space_info_no.hpp"


namespace boosting {

    AbstractBoostingRuleLearner::Config::Config()
        : AbstractRuleLearner::Config(BOOSTED_RULE_COMPARE_FUNCTION) {
        this->useCompleteHeads();
        this->useDenseStatistics();
        this->useNoL1Regularization();
        this->useNoL2Regularization();
        this->useLabelWiseLogisticLoss();
        this->useNoLabelBinning();
        this->useLabelWiseClassificationPredictor();
        this->useLabelWiseRegressionPredictor();
        this->useLabelWiseProbabilityPredictor();
    }

    std::unique_ptr<IHeadConfig>& AbstractBoostingRuleLearner::Config::getHeadConfigPtr() {
        return headConfigPtr_;
    }

    std::unique_ptr<IStatisticsConfig>& AbstractBoostingRuleLearner::Config::getStatisticsConfigPtr() {
        return statisticsConfigPtr_;
    }

    std::unique_ptr<IRegularizationConfig>& AbstractBoostingRuleLearner::Config::getL1RegularizationConfigPtr() {
        return l1RegularizationConfigPtr_;
    }

    std::unique_ptr<IRegularizationConfig>& AbstractBoostingRuleLearner::Config::getL2RegularizationConfigPtr() {
        return l2RegularizationConfigPtr_;
    }

    std::unique_ptr<ILossConfig>& AbstractBoostingRuleLearner::Config::getLossConfigPtr() {
        return lossConfigPtr_;
    }

    std::unique_ptr<ILabelBinningConfig>& AbstractBoostingRuleLearner::Config::getLabelBinningConfigPtr() {
        return labelBinningConfigPtr_;
    }

    std::unique_ptr<ILabelPredictorConfig>& AbstractBoostingRuleLearner::Config::getLabelPredictorConfigPtr() {
        return labelPredictorConfigPtr_;
    }

    // TODO Remove
    std::unique_ptr<IClassificationPredictorConfig>& AbstractBoostingRuleLearner::Config::getClassificationPredictorConfigPtr() {
        return classificationPredictorConfigPtr_;
    }

    std::unique_ptr<IScorePredictorConfig>& AbstractBoostingRuleLearner::Config::getScorePredictorConfigPtr() {
        return scorePredictorConfigPtr_;
    }

    // TODO Remove
    std::unique_ptr<IRegressionPredictorConfig>& AbstractBoostingRuleLearner::Config::getRegressionPredictorConfigPtr() {
        return regressionPredictorConfigPtr_;
    }

    std::unique_ptr<IProbabilityPredictorConfig>& AbstractBoostingRuleLearner::Config::getProbabilityPredictorConfigPtr() {
        return probabilityPredictorConfigPtr_;
    }

    // TODO Remove
    std::unique_ptr<IOldProbabilityPredictorConfig>& AbstractBoostingRuleLearner::Config::getOldProbabilityPredictorConfigPtr() {
        return oldProbabilityPredictorConfigPtr_;
    }

    void AbstractBoostingRuleLearner::Config::useCompleteHeads() {
        headConfigPtr_ = std::make_unique<CompleteHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useDenseStatistics() {
        statisticsConfigPtr_ = std::make_unique<DenseStatisticsConfig>(lossConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useNoL1Regularization() {
        l1RegularizationConfigPtr_ = std::make_unique<NoRegularizationConfig>();
    }

    void AbstractBoostingRuleLearner::Config::useNoL2Regularization() {
        l2RegularizationConfigPtr_ = std::make_unique<NoRegularizationConfig>();
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseLogisticLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseLogisticLossConfig>(headConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useNoLabelBinning() {
        labelBinningConfigPtr_ =
            std::make_unique<NoLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseClassificationPredictor() {
        // TODO Re-implement
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseRegressionPredictor() {
        // TODO Re-implement
        regressionPredictorConfigPtr_ =
            std::make_unique<LabelWiseRegressionPredictorConfig>(parallelPredictionConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        // TODO Re-implement
        oldProbabilityPredictorConfigPtr_ =
            std::make_unique<LabelWiseProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    AbstractBoostingRuleLearner::AbstractBoostingRuleLearner(IBoostingRuleLearner::IConfig& config,
                                                             Blas::DdotFunction ddotFunction,
                                                             Blas::DspmvFunction dspmvFunction,
                                                             Lapack::DsysvFunction dsysvFunction)
        : AbstractRuleLearner(config), config_(config), blas_(Blas(ddotFunction, dspmvFunction)),
          lapack_(Lapack(dsysvFunction)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AbstractBoostingRuleLearner::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getStatisticsConfigPtr()->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas_,
                                                                                 lapack_);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractBoostingRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<RuleListBuilderFactory>();
    }

    std::unique_ptr<ILabelPredictorFactory> AbstractBoostingRuleLearner::createLabelPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getLabelPredictorConfigPtr()->createPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<ISparseLabelPredictorFactory> AbstractBoostingRuleLearner::createSparseLabelPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        // TODO return config_.getLabelPredictorConfigPtr()->createSparsePredictorFactory(featureMatrix, numLabels);
        return nullptr;
    }

    // TODO Remove
    std::unique_ptr<IClassificationPredictorFactory> AbstractBoostingRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getClassificationPredictorConfigPtr()->createClassificationPredictorFactory(featureMatrix,
                                                                                                   numLabels);
    }

    std::unique_ptr<IScorePredictorFactory> AbstractBoostingRuleLearner::createScorePredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getScorePredictorConfigPtr()->createPredictorFactory(featureMatrix, numLabels);
    }

    // TODO Remove
    std::unique_ptr<IRegressionPredictorFactory> AbstractBoostingRuleLearner::createRegressionPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getRegressionPredictorConfigPtr()->createRegressionPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<IProbabilityPredictorFactory> AbstractBoostingRuleLearner::createProbabilityPredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getProbabilityPredictorConfigPtr()->createPredictorFactory(featureMatrix, numLabels);
    }

    // TODO Remove
    std::unique_ptr<IOldProbabilityPredictorFactory> AbstractBoostingRuleLearner::createOldProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getOldProbabilityPredictorConfigPtr()->createProbabilityPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<ILabelSpaceInfo> AbstractBoostingRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        // TODO Use correct configs
        if (config_.getClassificationPredictorConfigPtr()->isLabelVectorSetNeeded()
            || config_.getOldProbabilityPredictorConfigPtr()->isLabelVectorSetNeeded()
            || config_.getRegressionPredictorConfigPtr()->isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

}
