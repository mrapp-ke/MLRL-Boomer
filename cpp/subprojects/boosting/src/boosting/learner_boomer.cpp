#include "boosting/learner_boomer.hpp"

#include "boosting/binning/feature_binning_auto.hpp"
#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/prediction/predictor_binary_auto.hpp"
#include "boosting/prediction/predictor_probability_auto.hpp"
#include "boosting/rule_evaluation/head_type_auto.hpp"
#include "boosting/statistics/statistic_format_auto.hpp"

namespace boosting {

    Boomer::Config::Config() {
        this->useSequentialRuleModelAssemblage();
        this->useGreedyTopDownRuleInduction();
        this->useDefaultRule();
        this->useNoLabelSampling();
        this->useNoInstanceSampling();
        this->useFeatureSamplingWithoutReplacement();
        this->useParallelPrediction();
        this->useAutomaticDefaultRule();
        this->useAutomaticPartitionSampling();
        this->useAutomaticFeatureBinning();
        this->useSizeStoppingCriterion();
        this->useNoTimeStoppingCriterion();
        this->useNoRulePruning();
        this->useNoGlobalPruning();
        this->useNoSequentialPostOptimization();
        this->useConstantShrinkagePostProcessor();
        this->useAutomaticParallelRuleRefinement();
        this->useAutomaticParallelStatisticUpdate();
        this->useAutomaticHeads();
        this->useAutomaticStatistics();
        this->useLabelWiseLogisticLoss();
        this->useNoL1Regularization();
        this->useL2Regularization();
        this->useAutomaticLabelBinning();
        this->useAutomaticBinaryPredictor();
        this->useLabelWiseScorePredictor();
        this->useAutomaticProbabilityPredictor();
    }

    ISizeStoppingCriterionConfig& Boomer::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
        ref.setMaxRules(1000);
        return ref;
    }

    void Boomer::Config::useAutomaticFeatureBinning() {
        featureBinningConfigPtr_ = std::make_unique<AutomaticFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
    }

    void Boomer::Config::useAutomaticParallelRuleRefinement() {
        parallelRuleRefinementConfigPtr_ =
          std::make_unique<AutoParallelRuleRefinementConfig>(lossConfigPtr_, headConfigPtr_, featureSamplingConfigPtr_);
    }

    void Boomer::Config::useAutomaticParallelStatisticUpdate() {
        parallelStatisticUpdateConfigPtr_ = std::make_unique<AutoParallelStatisticUpdateConfig>(lossConfigPtr_);
    }

    void Boomer::Config::useAutomaticHeads() {
        headConfigPtr_ = std::make_unique<AutomaticHeadConfig>(lossConfigPtr_, labelBinningConfigPtr_,
                                                               parallelStatisticUpdateConfigPtr_,
                                                               l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    void Boomer::Config::useAutomaticStatistics() {
        statisticsConfigPtr_ =
          std::make_unique<AutomaticStatisticsConfig>(lossConfigPtr_, headConfigPtr_, defaultRuleConfigPtr_);
    }

    void Boomer::Config::useAutomaticLabelBinning() {
        labelBinningConfigPtr_ =
          std::make_unique<AutomaticLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    void Boomer::Config::useAutomaticBinaryPredictor() {
        binaryPredictorConfigPtr_ =
          std::make_unique<AutomaticBinaryPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void Boomer::Config::useAutomaticProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
          std::make_unique<AutomaticProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    Boomer::Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction)
        : AbstractBoostingRuleLearner(*configPtr, ddotFunction, dspmvFunction, dsysvFunction),
          configPtr_(std::move(configPtr)) {}

    std::unique_ptr<IBoomer::IConfig> createBoomerConfig() {
        return std::make_unique<Boomer::Config>();
    }

    std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                                          Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        return std::make_unique<Boomer>(std::move(configPtr), ddotFunction, dspmvFunction, dsysvFunction);
    }

}
