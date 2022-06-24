#include "boosting/learner.hpp"
#include "boosting/binning/label_binning_no.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/output/predictor_regression_label_wise.hpp"
#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/regularization_no.hpp"
#include "boosting/statistics/statistic_format_dense.hpp"
#include "common/output/label_space_info_no.hpp"


namespace boosting {

    AbstractBoostingRuleLearner::Config::Config() {
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

    const IHeadConfig& AbstractBoostingRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IStatisticsConfig& AbstractBoostingRuleLearner::Config::getStatisticsConfig() const {
        return *statisticsConfigPtr_;
    }

    const IRegularizationConfig& AbstractBoostingRuleLearner::Config::getL1RegularizationConfig() const {
        return *l1RegularizationConfigPtr_;
    }

    const IRegularizationConfig& AbstractBoostingRuleLearner::Config::getL2RegularizationConfig() const {
        return *l2RegularizationConfigPtr_;
    }

    const ILossConfig& AbstractBoostingRuleLearner::Config::getLossConfig() const {
        return *lossConfigPtr_;
    }

    const ILabelBinningConfig& AbstractBoostingRuleLearner::Config::getLabelBinningConfig() const {
        return *labelBinningConfigPtr_;
    }

    const IClassificationPredictorConfig& AbstractBoostingRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    const IRegressionPredictorConfig& AbstractBoostingRuleLearner::Config::getRegressionPredictorConfig() const {
        return *regressionPredictorConfigPtr_;
    }

    const IProbabilityPredictorConfig& AbstractBoostingRuleLearner::Config::getProbabilityPredictorConfig() const {
        return *probabilityPredictorConfigPtr_;
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
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseRegressionPredictor() {
        regressionPredictorConfigPtr_ =
            std::make_unique<LabelWiseRegressionPredictorConfig>(parallelPredictionConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<LabelWiseProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    AbstractBoostingRuleLearner::AbstractBoostingRuleLearner(const IBoostingRuleLearner::IConfig& config,
                                                             Blas::DdotFunction ddotFunction,
                                                             Blas::DspmvFunction dspmvFunction,
                                                             Lapack::DsysvFunction dsysvFunction)
        : AbstractRuleLearner(config), config_(config), blas_(Blas(ddotFunction, dspmvFunction)),
          lapack_(Lapack(dsysvFunction)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AbstractBoostingRuleLearner::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getStatisticsConfig().createStatisticsProviderFactory(featureMatrix, labelMatrix, blas_,
                                                                             lapack_);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractBoostingRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<RuleListBuilderFactory>();
    }

    std::unique_ptr<IClassificationPredictorFactory> AbstractBoostingRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getClassificationPredictorConfig().createClassificationPredictorFactory(featureMatrix,
                                                                                               numLabels);
    }

    std::unique_ptr<IRegressionPredictorFactory> AbstractBoostingRuleLearner::createRegressionPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getRegressionPredictorConfig().createRegressionPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<IProbabilityPredictorFactory> AbstractBoostingRuleLearner::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getProbabilityPredictorConfig().createProbabilityPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<ILabelSpaceInfo> AbstractBoostingRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (config_.getClassificationPredictorConfig().isLabelVectorSetNeeded()
            || config_.getProbabilityPredictorConfig().isLabelVectorSetNeeded()
            || config_.getRegressionPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

}
