#include "boosting/learner.hpp"

#include "boosting/binning/label_binning_no.hpp"
#include "boosting/losses/loss_label_wise_logistic.hpp"
#include "boosting/model/rule_list_builder.hpp"
#include "boosting/prediction/predictor_binary_label_wise.hpp"
#include "boosting/prediction/predictor_probability_label_wise.hpp"
#include "boosting/prediction/predictor_score_label_wise.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/regularization_no.hpp"
#include "boosting/rule_evaluation/rule_compare_function.hpp"
#include "boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    AbstractBoostingRuleLearner::Config::Config() : AbstractRuleLearner::Config(BOOSTED_RULE_COMPARE_FUNCTION) {
        this->useCompleteHeads();
        this->useDenseStatistics();
        this->useNoL1Regularization();
        this->useNoL2Regularization();
        this->useLabelWiseLogisticLoss();
        this->useNoLabelBinning();
        this->useLabelWiseBinaryPredictor();
        this->useLabelWiseScorePredictor();
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

    void AbstractBoostingRuleLearner::Config::useCompleteHeads() {
        headConfigPtr_ = std::make_unique<CompleteHeadConfig>(labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_,
                                                              l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
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

    ILabelWiseBinaryPredictorConfig& AbstractBoostingRuleLearner::Config::useLabelWiseBinaryPredictor() {
        std::unique_ptr<LabelWiseBinaryPredictorConfig> ptr =
          std::make_unique<LabelWiseBinaryPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
        ILabelWiseBinaryPredictorConfig& ref = *ptr;
        binaryPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseScorePredictor() {
        scorePredictorConfigPtr_ = std::make_unique<LabelWiseScorePredictorConfig>(parallelPredictionConfigPtr_);
    }

    void AbstractBoostingRuleLearner::Config::useLabelWiseProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
          std::make_unique<LabelWiseProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    AbstractBoostingRuleLearner::AbstractBoostingRuleLearner(IBoostingRuleLearner::IConfig& config,
                                                             Blas::DdotFunction ddotFunction,
                                                             Blas::DspmvFunction dspmvFunction,
                                                             Lapack::DsysvFunction dsysvFunction)
        : AbstractRuleLearner(config), config_(config), blas_(Blas(ddotFunction, dspmvFunction)),
          lapack_(Lapack(dsysvFunction)) {}

    std::unique_ptr<IStatisticsProviderFactory> AbstractBoostingRuleLearner::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getStatisticsConfigPtr()->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas_,
                                                                                 lapack_);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractBoostingRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<RuleListBuilderFactory>();
    }

}
