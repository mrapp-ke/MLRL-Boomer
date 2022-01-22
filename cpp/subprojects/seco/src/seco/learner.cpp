#include "seco/learner.hpp"
#include "seco/model/decision_list_builder.hpp"


namespace seco {

    SeCoRuleLearner::Config::Config() {
        this->useSizeStoppingCriterion();
        this->useLabelWiseStratifiedInstanceSampling();
        this->useIrepPruning();
        this->useParallelRuleRefinement();
        this->useCoverageStoppingCriterion();
        this->useSingleLabelHeads();
        this->useFMeasureHeuristic();
        this->useAccuracyPruningHeuristic();
        this->usePeakLiftFunction();
        this->useLabelWiseClassificationPredictor();
    }

    const CoverageStoppingCriterionConfig* SeCoRuleLearner::Config::getCoverageStoppingCriterionConfig() const {
        return coverageStoppingCriterionConfigPtr_.get();
    }

    const IHeadConfig& SeCoRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
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

    ISingleLabelHeadConfig& SeCoRuleLearner::Config::useSingleLabelHeads() {
        std::unique_ptr<SingleLabelHeadConfig> ptr =
            std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_);
        ISingleLabelHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    IPartialHeadConfig& SeCoRuleLearner::Config::usePartialHeads() {
        std::unique_ptr<PartialHeadConfig> ptr =
            std::make_unique<PartialHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_,
                                                liftFunctionConfigPtr_);
        IPartialHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    IAccuracyConfig& SeCoRuleLearner::Config::useAccuracyHeuristic() {
        std::unique_ptr<AccuracyConfig> ptr = std::make_unique<AccuracyConfig>();
        IAccuracyConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IFMeasureConfig& SeCoRuleLearner::Config::useFMeasureHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILaplaceConfig& SeCoRuleLearner::Config::useLaplaceHeuristic() {
        std::unique_ptr<LaplaceConfig> ptr = std::make_unique<LaplaceConfig>();
        ILaplaceConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IMEstimateConfig& SeCoRuleLearner::Config::useMEstimateHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IPrecisionConfig& SeCoRuleLearner::Config::usePrecisionHeuristic() {
        std::unique_ptr<PrecisionConfig> ptr = std::make_unique<PrecisionConfig>();
        IPrecisionConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IRecallConfig& SeCoRuleLearner::Config::useRecallHeuristic() {
        std::unique_ptr<RecallConfig> ptr = std::make_unique<RecallConfig>();
        IRecallConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IWraConfig& SeCoRuleLearner::Config::useWraHeuristic() {
        std::unique_ptr<WraConfig> ptr = std::make_unique<WraConfig>();
        IWraConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IAccuracyConfig& SeCoRuleLearner::Config::useAccuracyPruningHeuristic() {
        std::unique_ptr<AccuracyConfig> ptr = std::make_unique<AccuracyConfig>();
        IAccuracyConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IFMeasureConfig& SeCoRuleLearner::Config::useFMeasurePruningHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILaplaceConfig& SeCoRuleLearner::Config::useLaplacePruningHeuristic() {
        std::unique_ptr<LaplaceConfig> ptr = std::make_unique<LaplaceConfig>();
        ILaplaceConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IMEstimateConfig& SeCoRuleLearner::Config::useMEstimatePruningHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IPrecisionConfig& SeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        std::unique_ptr<PrecisionConfig> ptr = std::make_unique<PrecisionConfig>();
        IPrecisionConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IRecallConfig& SeCoRuleLearner::Config::useRecallPruningHeuristic() {
        std::unique_ptr<RecallConfig> ptr = std::make_unique<RecallConfig>();
        IRecallConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IWraConfig& SeCoRuleLearner::Config::useWraPruningHeuristic() {
        std::unique_ptr<WraConfig> ptr = std::make_unique<WraConfig>();
        IWraConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    IPeakLiftFunctionConfig& SeCoRuleLearner::Config::usePeakLiftFunction() {
        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
        IPeakLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseClassificationPredictorConfig& SeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        std::unique_ptr<LabelWiseClassificationPredictorConfig> ptr =
            std::make_unique<LabelWiseClassificationPredictorConfig>();
        ILabelWiseClassificationPredictorConfig& ref = *ptr;
        classificationPredictorConfigPtr_ = std::move(ptr);
        return ref;
    }

    SeCoRuleLearner::SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IStoppingCriterionFactory> SeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = configPtr_->getCoverageStoppingCriterionConfig();
        return config ? config->configure() : nullptr;
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

    std::unique_ptr<IStatisticsProviderFactory> SeCoRuleLearner::createStatisticsProviderFactory(
            const IRowWiseLabelMatrix& labelMatrix) const {
        return configPtr_->getHeadConfig().configure(labelMatrix);
    }

    std::unique_ptr<IModelBuilder> SeCoRuleLearner::createModelBuilder() const {
        return std::make_unique<DecisionListBuilder>();
    }

    std::unique_ptr<IClassificationPredictorFactory> SeCoRuleLearner::createClassificationPredictorFactory() const {
        return configPtr_->getClassificationPredictorConfig().configure();
    }

    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig() {
        return std::make_unique<SeCoRuleLearner::Config>();
    }

    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<SeCoRuleLearner>(std::move(configPtr));
    }

}
