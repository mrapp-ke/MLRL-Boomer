#include "seco/learner_seco.hpp"
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
#include "common/pruning/pruning_irep.hpp"


namespace seco {

    MultiLabelSeCoRuleLearner::Config::Config() {
        this->useParallelPrediction();
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

    const CoverageStoppingCriterionConfig* MultiLabelSeCoRuleLearner::Config::getCoverageStoppingCriterionConfig() const {
        return coverageStoppingCriterionConfigPtr_.get();
    }

    const IHeadConfig& MultiLabelSeCoRuleLearner::Config::getHeadConfig() const {
        return *headConfigPtr_;
    }

    const IHeuristicConfig& MultiLabelSeCoRuleLearner::Config::getHeuristicConfig() const {
        return *heuristicConfigPtr_;
    }

    const IHeuristicConfig& MultiLabelSeCoRuleLearner::Config::getPruningHeuristicConfig() const {
        return *pruningHeuristicConfigPtr_;
    }

    const ILiftFunctionConfig& MultiLabelSeCoRuleLearner::Config::getLiftFunctionConfig() const {
        return *liftFunctionConfigPtr_;
    }

    const IClassificationPredictorConfig& MultiLabelSeCoRuleLearner::Config::getClassificationPredictorConfig() const {
        return *classificationPredictorConfigPtr_;
    }

    IGreedyTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useGreedyTopDownRuleInduction() {
        IGreedyTopDownRuleInductionConfig& config = AbstractRuleLearner::Config::useGreedyTopDownRuleInduction();
        config.setRecalculatePredictions(false);
        return config;
    }

    IBeamSearchTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
            std::make_unique<BeamSearchTopDownRuleInductionConfig>(parallelRuleRefinementConfigPtr_);
        IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
        ruleInductionConfigPtr_ = std::move(ptr);
        ref.setRecalculatePredictions(false);
        return ref;
    }

    IEqualWidthFeatureBinningConfig& MultiLabelSeCoRuleLearner::Config::useEqualWidthFeatureBinning() {
        std::unique_ptr<EqualWidthFeatureBinningConfig> ptr =
            std::make_unique<EqualWidthFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
        IEqualWidthFeatureBinningConfig& ref = *ptr;
        featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualFrequencyFeatureBinningConfig& MultiLabelSeCoRuleLearner::Config::useEqualFrequencyFeatureBinning() {
        std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
            std::make_unique<EqualFrequencyFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
        IEqualFrequencyFeatureBinningConfig& ref = *ptr;
        featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelSamplingWithoutReplacementConfig& MultiLabelSeCoRuleLearner::Config::useLabelSamplingWithoutReplacement() {
        std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
            std::make_unique<LabelSamplingWithoutReplacementConfig>();
        ILabelSamplingWithoutReplacementConfig& ref = *ptr;
        labelSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithReplacementConfig& MultiLabelSeCoRuleLearner::Config::useInstanceSamplingWithReplacement() {
        std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithReplacementConfig>();
        IInstanceSamplingWithReplacementConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithoutReplacementConfig& MultiLabelSeCoRuleLearner::Config::useInstanceSamplingWithoutReplacement() {
        std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithoutReplacementConfig>();
        IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseStratifiedInstanceSamplingConfig& MultiLabelSeCoRuleLearner::Config::useLabelWiseStratifiedInstanceSampling() {
        std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
        ILabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseStratifiedInstanceSamplingConfig& MultiLabelSeCoRuleLearner::Config::useExampleWiseStratifiedInstanceSampling() {
        std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
        IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IFeatureSamplingWithoutReplacementConfig& MultiLabelSeCoRuleLearner::Config::useFeatureSamplingWithoutReplacement() {
        std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
            std::make_unique<FeatureSamplingWithoutReplacementConfig>();
        IFeatureSamplingWithoutReplacementConfig& ref = *ptr;
        featureSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IRandomBiPartitionSamplingConfig& MultiLabelSeCoRuleLearner::Config::useRandomBiPartitionSampling() {
        std::unique_ptr<RandomBiPartitionSamplingConfig> ptr = std::make_unique<RandomBiPartitionSamplingConfig>();
        IRandomBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseStratifiedBiPartitionSamplingConfig& MultiLabelSeCoRuleLearner::Config::useLabelWiseStratifiedBiPartitionSampling() {
        std::unique_ptr<LabelWiseStratifiedBiPartitionSamplingConfig> ptr =
            std::make_unique<LabelWiseStratifiedBiPartitionSamplingConfig>();
        ILabelWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseStratifiedBiPartitionSamplingConfig& MultiLabelSeCoRuleLearner::Config::useExampleWiseStratifiedBiPartitionSampling() {
        std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
            std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
        IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useIrepPruning() {
        pruningConfigPtr_ = std::make_unique<IrepConfig>();
    }

    IManualMultiThreadingConfig& MultiLabelSeCoRuleLearner::Config::useParallelRuleRefinement() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelRuleRefinementConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualMultiThreadingConfig& MultiLabelSeCoRuleLearner::Config::useParallelStatisticUpdate() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelStatisticUpdateConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualMultiThreadingConfig& MultiLabelSeCoRuleLearner::Config::useParallelPrediction() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelPredictionConfigPtr_ = std::move(ptr);
        return ref;
    }

    ISizeStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useSizeStoppingCriterion() {
        std::unique_ptr<SizeStoppingCriterionConfig> ptr = std::make_unique<SizeStoppingCriterionConfig>();
        ISizeStoppingCriterionConfig& ref = *ptr;
        sizeStoppingCriterionConfigPtr_ = std::move(ptr);
        ref.setMaxRules(500);
        return ref;
    }

    ITimeStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useTimeStoppingCriterion() {
        std::unique_ptr<TimeStoppingCriterionConfig> ptr = std::make_unique<TimeStoppingCriterionConfig>();
        ITimeStoppingCriterionConfig& ref = *ptr;
        timeStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IMeasureStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useMeasureStoppingCriterion() {
        std::unique_ptr<MeasureStoppingCriterionConfig> ptr = std::make_unique<MeasureStoppingCriterionConfig>();
        IMeasureStoppingCriterionConfig& ref = *ptr;
        measureStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useNoCoverageStoppingCriterion() {
        coverageStoppingCriterionConfigPtr_ = nullptr;
    }

    ICoverageStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useCoverageStoppingCriterion() {
        std::unique_ptr<CoverageStoppingCriterionConfig> ptr = std::make_unique<CoverageStoppingCriterionConfig>();
        ICoverageStoppingCriterionConfig& ref = *ptr;
        coverageStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_);
    }

    void MultiLabelSeCoRuleLearner::Config::usePartialHeads() {
        headConfigPtr_ = std::make_unique<PartialHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_,
                                                             liftFunctionConfigPtr_);
    }

    void MultiLabelSeCoRuleLearner::Config::useAccuracyHeuristic() {
        heuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& MultiLabelSeCoRuleLearner::Config::useFMeasureHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useLaplaceHeuristic() {
        heuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& MultiLabelSeCoRuleLearner::Config::useMEstimateHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        heuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::usePrecisionHeuristic() {
        heuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useRecallHeuristic() {
        heuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useWraHeuristic() {
        heuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useAccuracyPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<AccuracyConfig>();
    }

    IFMeasureConfig& MultiLabelSeCoRuleLearner::Config::useFMeasurePruningHeuristic() {
        std::unique_ptr<FMeasureConfig> ptr = std::make_unique<FMeasureConfig>();
        IFMeasureConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useLaplacePruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<LaplaceConfig>();
    }

    IMEstimateConfig& MultiLabelSeCoRuleLearner::Config::useMEstimatePruningHeuristic() {
        std::unique_ptr<MEstimateConfig> ptr = std::make_unique<MEstimateConfig>();
        IMEstimateConfig& ref = *ptr;
        pruningHeuristicConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::usePrecisionPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<PrecisionConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useRecallPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useWraPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    IPeakLiftFunctionConfig& MultiLabelSeCoRuleLearner::Config::usePeakLiftFunction() {
        std::unique_ptr<PeakLiftFunctionConfig> ptr = std::make_unique<PeakLiftFunctionConfig>();
        IPeakLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IKlnLiftFunctionConfig& MultiLabelSeCoRuleLearner::Config::useKlnLiftFunction() {
        std::unique_ptr<KlnLiftFunctionConfig> ptr = std::make_unique<KlnLiftFunctionConfig>();
        IKlnLiftFunctionConfig& ref = *ptr;
        liftFunctionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::useLabelWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<LabelWiseClassificationPredictorConfig>(parallelPredictionConfigPtr_);
    }

    MultiLabelSeCoRuleLearner::MultiLabelSeCoRuleLearner(std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr)
        : AbstractRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IStoppingCriterionFactory> MultiLabelSeCoRuleLearner::createCoverageStoppingCriterionFactory() const {
        const CoverageStoppingCriterionConfig* config = configPtr_->getCoverageStoppingCriterionConfig();
        return config ? config->createStoppingCriterionFactory() : nullptr;
    }

    void MultiLabelSeCoRuleLearner::createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
        AbstractRuleLearner::createStoppingCriterionFactories(factory);
        std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
            this->createCoverageStoppingCriterionFactory();

        if (stoppingCriterionFactory) {
            factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
        }
    }

    std::unique_ptr<IStatisticsProviderFactory> MultiLabelSeCoRuleLearner::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return configPtr_->getHeadConfig().createStatisticsProviderFactory(labelMatrix);
    }

    std::unique_ptr<IModelBuilderFactory> MultiLabelSeCoRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<DecisionListBuilderFactory>();
    }

    std::unique_ptr<ILabelSpaceInfo> MultiLabelSeCoRuleLearner::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (configPtr_->getClassificationPredictorConfig().isLabelVectorSetNeeded()) {
            return createLabelVectorSet(labelMatrix);
        } else {
            return createNoLabelSpaceInfo();
        }
    }

    std::unique_ptr<IClassificationPredictorFactory> MultiLabelSeCoRuleLearner::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        return configPtr_->getClassificationPredictorConfig()
            .createClassificationPredictorFactory(featureMatrix, numLabels);
    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> createMultiLabelSeCoRuleLearnerConfig() {
        return std::make_unique<MultiLabelSeCoRuleLearner::Config>();
    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner> createMultiLabelSeCoRuleLearner(std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<MultiLabelSeCoRuleLearner>(std::move(configPtr));
    }

}
