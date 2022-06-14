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
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/output/label_space_info_no.hpp"


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

    IGreedyTopDownRuleInductionConfig& SeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& config = AbstractRuleLearner::Config::useGreedyTopDownRuleInduction();
        config.setRecalculatePredictions(false);
        return config;
    }

    IBeamSearchTopDownRuleInductionConfig& SeCoRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
            std::make_unique<BeamSearchTopDownRuleInductionConfig>(this->parallelRuleRefinementConfigPtr_);
        IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
        this->ruleInductionConfigPtr_ = std::move(ptr);
        ref.setRecalculatePredictions(false);
        return ref;
    }

    IEqualWidthFeatureBinningConfig& SeCoRuleLearner::Config::useEqualWidthFeatureBinning() {
        std::unique_ptr<EqualWidthFeatureBinningConfig> ptr =
            std::make_unique<EqualWidthFeatureBinningConfig>(this->parallelStatisticUpdateConfigPtr_);
        IEqualWidthFeatureBinningConfig& ref = *ptr;
        this->featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualFrequencyFeatureBinningConfig& SeCoRuleLearner::Config::useEqualFrequencyFeatureBinning() {
        std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
            std::make_unique<EqualFrequencyFeatureBinningConfig>(this->parallelStatisticUpdateConfigPtr_);
        IEqualFrequencyFeatureBinningConfig& ref = *ptr;
        this->featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelSamplingWithoutReplacementConfig& SeCoRuleLearner::Config::useLabelSamplingWithoutReplacement() {
        std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
            std::make_unique<LabelSamplingWithoutReplacementConfig>();
        ILabelSamplingWithoutReplacementConfig& ref = *ptr;
        this->labelSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ISizeStoppingCriterionConfig& SeCoRuleLearner::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = AbstractRuleLearner::Config::useSizeStoppingCriterion();
        ref.setMaxRules(500);
        return ref;
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

    void SeCoRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_);
    }

    void SeCoRuleLearner::Config::usePartialHeads() {
        headConfigPtr_ = std::make_unique<PartialHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_,
                                                             liftFunctionConfigPtr_);
    }

    void SeCoRuleLearner::Config::useAccuracyHeuristic() {
        heuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& SeCoRuleLearner::Config::useFMeasureHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void SeCoRuleLearner::Config::useLaplaceHeuristic() {
        heuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& SeCoRuleLearner::Config::useMEstimateHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void SeCoRuleLearner::Config::usePrecisionHeuristic() {
        heuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void SeCoRuleLearner::Config::useRecallHeuristic() {
        heuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void SeCoRuleLearner::Config::useWraHeuristic() {
        heuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    void SeCoRuleLearner::Config::useAccuracyPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& SeCoRuleLearner::Config::useFMeasurePruningHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void SeCoRuleLearner::Config::useLaplacePruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& SeCoRuleLearner::Config::useMEstimatePruningHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void SeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void SeCoRuleLearner::Config::useRecallPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void SeCoRuleLearner::Config::useWraPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    IPeakLiftFunctionConfig& SeCoRuleLearner::Config::usePeakLiftFunction() {
        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
        IPeakLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IKlnLiftFunctionConfig& SeCoRuleLearner::Config::useKlnLiftFunction() {
        std::unique_ptr<KlnLiftFunctionConfig> ptr = std::make_unique<KlnLiftFunctionConfig>();
        IKlnLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void SeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(parallelPredictionConfigPtr_);
    }

    SeCoRuleLearner::SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IStoppingCriterionFactory> SeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = configPtr_->getCoverageStoppingCriterionConfig();
        return config ? config->createStoppingCriterionFactory() : nullptr;
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
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return configPtr_->getHeadConfig().createStatisticsProviderFactory(labelMatrix);
    }

    std::unique_ptr<IModelBuilderFactory> SeCoRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<DecisionListBuilderFactory>();
    }

    std::unique_ptr<ILabelSpaceInfo> SeCoRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (configPtr_->getClassificationPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

    std::unique_ptr<IClassificationPredictorFactory> SeCoRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return configPtr_->getClassificationPredictorConfig()
            .createClassificationPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig() {
        return std::make_unique<SeCoRuleLearner::Config>();
    }

    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<SeCoRuleLearner>(std::move(configPtr));
    }

}
