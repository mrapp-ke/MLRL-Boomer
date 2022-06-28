#include "seco/learner.hpp"
#include "seco/heuristics/heuristic_accuracy.hpp"
#include "seco/heuristics/heuristic_laplace.hpp"
#include "seco/heuristics/heuristic_precision.hpp"
#include "seco/heuristics/heuristic_recall.hpp"
#include "seco/heuristics/heuristic_wra.hpp"
#include "seco/model/decision_list_builder.hpp"
#include "seco/output/predictor_classification_label_wise.hpp"
#include "seco/rule_evaluation/head_type_partial.hpp"
#include "seco/rule_evaluation/head_type_single.hpp"
#include "common/output/label_space_info_no.hpp"


namespace seco {

    AbstractSeCoRuleLearner::Config::Config() {
        this->useCoverageStoppingCriterion();
        this->useSingleLabelHeads();
        this->usePrecisionHeuristic();
        this->usePrecisionPruningHeuristic();
        this->usePeakLiftFunction();
        this->useLabelWiseClassificationPredictor();
    }

    const CoverageStoppingCriterionConfig* AbstractSeCoRuleLearner::Config::getCoverageStoppingCriterionConfig() const {
        return coverageStoppingCriterionConfigPtr_.get();
    }

    const IHeadConfig& AbstractSeCoRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IHeuristicConfig& AbstractSeCoRuleLearner::Config::getHeuristicConfig() const {
        return *heuristicConfigPtr_;
    }

    const IHeuristicConfig& AbstractSeCoRuleLearner::Config::getPruningHeuristicConfig() const {
        return *pruningHeuristicConfigPtr_;
    }

    const ILiftFunctionConfig& AbstractSeCoRuleLearner::Config::getLiftFunctionConfig() const {
        return *liftFunctionConfigPtr_;
    }

    const IClassificationPredictorConfig& AbstractSeCoRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    IGreedyTopDownRuleInductionConfig& AbstractSeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& config = AbstractRuleLearner::Config::useGreedyTopDownRuleInduction();
        config.setRecalculatePredictions(false);
        return config;
    }

    void AbstractSeCoRuleLearner::Config::useNoCoverageStoppingCriterion() {
        coverageStoppingCriterionConfigPtr_ = nullptr;
    }

    ICoverageStoppingCriterionConfig& AbstractSeCoRuleLearner::Config::useCoverageStoppingCriterion() {
        std::unique_ptr<CoverageStoppingCriterionConfig> ptr = std::make_unique<CoverageStoppingCriterionConfig>();
        ICoverageStoppingCriterionConfig& ref = *ptr;
        coverageStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_);
    }

    void AbstractSeCoRuleLearner::Config::usePartialHeads() {
        headConfigPtr_ = std::make_unique<PartialHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_,
                                                             liftFunctionConfigPtr_);
    }

    void AbstractSeCoRuleLearner::Config::useAccuracyHeuristic() {
        heuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& AbstractSeCoRuleLearner::Config::useFMeasureHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::useLaplaceHeuristic() {
        heuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& AbstractSeCoRuleLearner::Config::useMEstimateHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::usePrecisionHeuristic() {
        heuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useRecallHeuristic() {
        heuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useWraHeuristic() {
        heuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useAccuracyPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& AbstractSeCoRuleLearner::Config::useFMeasurePruningHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::useLaplacePruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& AbstractSeCoRuleLearner::Config::useMEstimatePruningHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useRecallPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void AbstractSeCoRuleLearner::Config::useWraPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    IPeakLiftFunctionConfig& AbstractSeCoRuleLearner::Config::usePeakLiftFunction() {
        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
        IPeakLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IKlnLiftFunctionConfig& AbstractSeCoRuleLearner::Config::useKlnLiftFunction() {
        std::unique_ptr<KlnLiftFunctionConfig> ptr = std::make_unique<KlnLiftFunctionConfig>();
        IKlnLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void AbstractSeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(parallelPredictionConfigPtr_);
    }

    AbstractSeCoRuleLearner::AbstractSeCoRuleLearner(ISeCoRuleLearner::IConfig& config)
        : AbstractRuleLearner(config), config_(config) {

    }

    std::unique_ptr<IStoppingCriterionFactory> AbstractSeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = config_.getCoverageStoppingCriterionConfig();
        return config ? config->createStoppingCriterionFactory() : nullptr;
    }

    void AbstractSeCoRuleLearner::createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
        AbstractRuleLearner::createStoppingCriterionFactories(factory);
        std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
            this->createCoverageStoppingCriterionFactory();

        if (stoppingCriterionFactory) {
            factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> AbstractSeCoRuleLearner::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getHeadConfig().createStatisticsProviderFactory(labelMatrix);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractSeCoRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<DecisionListBuilderFactory>();
    }

    std::unique_ptr<ILabelSpaceInfo> AbstractSeCoRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (config_.getClassificationPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

    std::unique_ptr<IClassificationPredictorFactory> AbstractSeCoRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return config_.getClassificationPredictorConfig().createClassificationPredictorFactory(featureMatrix,
                                                                                               numLabels);
    }

}
