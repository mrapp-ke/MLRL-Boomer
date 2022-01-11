#include "seco/learner.hpp"


namespace seco {

    SeCoRuleLearner::Config::Config()
        : heuristicConfigPtr_(std::make_unique<FMeasureConfig>()) {

    }

    const IHeuristicConfig& SeCoRuleLearner::Config::getHeuristicConfig() const {
        return *heuristicConfigPtr_;
    }

    AccuracyConfig& SeCoRuleLearner::Config::useAccuracyHeuristic() {
        std::unique_ptr<AccuracyConfig> ptr = std::make_unique<AccuracyConfig>();
        AccuracyConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    FMeasureConfig& SeCoRuleLearner::Config::useFMeasureHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        FMeasureConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    LaplaceConfig& SeCoRuleLearner::Config::useLaplaceHeuristic() {
        std::unique_ptr<LaplaceConfig> ptr = std::make_unique<LaplaceConfig>();
        LaplaceConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    MEstimateConfig& SeCoRuleLearner::Config::useMEstimateHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        MEstimateConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    PrecisionConfig& SeCoRuleLearner::Config::usePrecisionHeuristic() {
        std::unique_ptr<PrecisionConfig> ptr = std::make_unique<PrecisionConfig>();
        PrecisionConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    RecallConfig& SeCoRuleLearner::Config::useRecallHeuristic() {
        std::unique_ptr<RecallConfig> ptr = std::make_unique<RecallConfig>();
        RecallConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    WraConfig& SeCoRuleLearner::Config::useWraHeuristic() {
        std::unique_ptr<WraConfig> ptr = std::make_unique<WraConfig>();
        WraConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    std::unique_ptr<IStatisticsProviderFactory> SeCoRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IModelBuilder> SeCoRuleLearner::createModelBuilder() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IClassificationPredictorFactory> SeCoRuleLearner::createClassificationPredictorFactory() const {
        // TODO Implement
        return nullptr;
    }

    SeCoRuleLearner::SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(std::move(configPtr)) {

    }

    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig() {
        return std::make_unique<SeCoRuleLearner::Config>();
    }

    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<SeCoRuleLearner>(std::move(configPtr));
    }

}
