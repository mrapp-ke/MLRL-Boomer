#include "seco/learner.hpp"
#include "seco/model/decision_list_builder.hpp"


namespace seco {

    SeCoRuleLearner::Config::Config() {
        this->useSizeStoppingCriterion();
        this->useLabelWiseStratifiedInstanceSampling();
        this->useIrepPruning();
        this->useCoverageStoppingCriterion();
        this->useFMeasureHeuristic();
        this->useAccuracyPruningHeuristic();
        this->usePeakLiftFunction();
        this->useLabelWiseClassificationPredictor();
    }

    const CoverageStoppingCriterionConfig* SeCoRuleLearner::Config::getCoverageStoppingCriterionConfig() const {
        return coverageStoppingCriterionConfigPtr_.get();
    }

    const IHeuristicConfig& SeCoRuleLearner::Config::getHeuristicConfig() const {
        return *heuristicConfigPtr_;
    }

    const IHeuristicConfig& SeCoRuleLearner::Config::getPruningHeuristicConfig() const {
        return *pruningHeuristicConfigPtr_;
    }

    const ILiftFunctionConfig& SeCoRuleLearner::Config::getLiftFunctionConfig() const {
        return *liftFunctionConfigPtr_;
    }

    const IClassificationPredictorConfig& SeCoRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    ISizeStoppingCriterionConfig& SeCoRuleLearner::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = AbstractRuleLearner::Config::useSizeStoppingCriterion();
        ref.setMaxRules(500);
        return ref;
    }

    ITopDownRuleInductionConfig& SeCoRuleLearner::Config::useTopDownRuleInduction() {
        ITopDownRuleInductionConfig& config = AbstractRuleLearner::Config::useTopDownRuleInduction();
        config.setRecalculatePredictions(false);
        return config;
    }

    void SeCoRuleLearner::Config::useNoCoverageStoppingCriterion() {
        coverageStoppingCriterionConfigPtr_ = nullptr;
    }

    ICoverageStoppingCriterionConfig& SeCoRuleLearner::Config::useCoverageStoppingCriterion() {
        std::unique_ptr<CoverageStoppingCriterionConfig> ptr = std::make_unique<CoverageStoppingCriterionConfig>();
        ICoverageStoppingCriterionConfig& ref = *ptr;
        coverageStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
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

    AccuracyConfig& SeCoRuleLearner::Config::useAccuracyPruningHeuristic() {
        std::unique_ptr<AccuracyConfig> ptr = std::make_unique<AccuracyConfig>();
        AccuracyConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    FMeasureConfig& SeCoRuleLearner::Config::useFMeasurePruningHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        FMeasureConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    LaplaceConfig& SeCoRuleLearner::Config::useLaplacePruningHeuristic() {
        std::unique_ptr<LaplaceConfig> ptr = std::make_unique<LaplaceConfig>();
        LaplaceConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    MEstimateConfig& SeCoRuleLearner::Config::useMEstimatePruningHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        MEstimateConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    PrecisionConfig& SeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        std::unique_ptr<PrecisionConfig> ptr = std::make_unique<PrecisionConfig>();
        PrecisionConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    RecallConfig& SeCoRuleLearner::Config::useRecallPruningHeuristic() {
        std::unique_ptr<RecallConfig> ptr = std::make_unique<RecallConfig>();
        RecallConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    WraConfig& SeCoRuleLearner::Config::useWraPruningHeuristic() {
        std::unique_ptr<WraConfig> ptr = std::make_unique<WraConfig>();
        WraConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    PeakLiftFunctionConfig& SeCoRuleLearner::Config::usePeakLiftFunction() {
        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
        PeakLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    LabelWiseClassificationPredictorConfig& SeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        std::unique_ptr<LabelWiseClassificationPredictorConfig> ptr =
            std::make_unique<LabelWiseClassificationPredictorConfig>();
        LabelWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    SeCoRuleLearner::SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IHeuristicFactory> SeCoRuleLearner::createHeuristicFactory() const {
        const IHeuristicConfig* baseConfig = &configPtr_->getHeuristicConfig();

        if (dynamic_cast<const AccuracyConfig*>(baseConfig)) {
            return std::make_unique<AccuracyFactory>();
        } else if (auto* config = dynamic_cast<const FMeasureConfig*>(baseConfig)) {
            return std::make_unique<FMeasureFactory>(config->getBeta());
        } else if (dynamic_cast<const LaplaceConfig*>(baseConfig)) {
            return std::make_unique<LaplaceFactory>();
        } else if (auto* config = dynamic_cast<const MEstimateConfig*>(baseConfig)) {
            return std::make_unique<MEstimateFactory>(config->getM());
        } else if (dynamic_cast<const PrecisionConfig*>(baseConfig)) {
            return std::make_unique<PrecisionFactory>();
        } else if (dynamic_cast<const RecallConfig*>(baseConfig)) {
            return std::make_unique<RecallFactory>();
        } else if (dynamic_cast<const WraConfig*>(baseConfig)) {
            return std::make_unique<WraFactory>();
        }

        throw std::runtime_error("Failed to create IHeuristicFactory");
    }

    // TODO Avoid redundant code
    std::unique_ptr<IHeuristicFactory> SeCoRuleLearner::createPruningHeuristicFactory() const {
        const IHeuristicConfig* baseConfig = &configPtr_->getPruningHeuristicConfig();

        if (dynamic_cast<const AccuracyConfig*>(baseConfig)) {
            return std::make_unique<AccuracyFactory>();
        } else if (auto* config = dynamic_cast<const FMeasureConfig*>(baseConfig)) {
            return std::make_unique<FMeasureFactory>(config->getBeta());
        } else if (dynamic_cast<const LaplaceConfig*>(baseConfig)) {
            return std::make_unique<LaplaceFactory>();
        } else if (auto* config = dynamic_cast<const MEstimateConfig*>(baseConfig)) {
            return std::make_unique<MEstimateFactory>(config->getM());
        } else if (dynamic_cast<const PrecisionConfig*>(baseConfig)) {
            return std::make_unique<PrecisionFactory>();
        } else if (dynamic_cast<const RecallConfig*>(baseConfig)) {
            return std::make_unique<RecallFactory>();
        } else if (dynamic_cast<const WraConfig*>(baseConfig)) {
            return std::make_unique<WraFactory>();
        }

        throw std::runtime_error("Failed to create IHeuristicFactory");
    }

    std::unique_ptr<ILiftFunctionFactory> SeCoRuleLearner::createLiftFunctionFactory() const {
        const ILiftFunctionConfig* baseConfig = &configPtr_->getLiftFunctionConfig();

        if (auto* config = dynamic_cast<const PeakLiftFunctionConfig*>(baseConfig)) {
            return std::make_unique<PeakLiftFunctionFactory>(config->getNumLabels(), config->getPeakLabel(),
                                                             config->getMaxLift(), config->getCurvature());
        }

        throw std::runtime_error("Failed to create ILiftFunctionFactory");
    }

    std::unique_ptr<IStoppingCriterionFactory> SeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = configPtr_->getCoverageStoppingCriterionConfig();
        return config ? config->create() : nullptr;
    }

    void SeCoRuleLearner::createStoppingCriterionFactories(
            std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const {
        AbstractRuleLearner::createStoppingCriterionFactories(stoppingCriterionFactories);
        std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
            this->createCoverageStoppingCriterionFactory();

        if (stoppingCriterionFactory) {
            stoppingCriterionFactories.push_front(std::move(stoppingCriterionFactory));
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> SeCoRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IModelBuilder> SeCoRuleLearner::createModelBuilder() const {
        return std::make_unique<DecisionListBuilder>();
    }

    std::unique_ptr<IClassificationPredictorFactory> SeCoRuleLearner::createClassificationPredictorFactory() const {
        const IClassificationPredictorConfig* baseConfig = &configPtr_->getClassificationPredictorConfig();

        if (auto* config = dynamic_cast<const LabelWiseClassificationPredictorConfig*>(baseConfig)) {
            return std::make_unique<LabelWiseClassificationPredictorFactory>(config->getNumThreads());
        }

        throw std::runtime_error("Failed to create IClassificationPredictorFactory");
    }

    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig() {
        return std::make_unique<SeCoRuleLearner::Config>();
    }

    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<SeCoRuleLearner>(std::move(configPtr));
    }

}
