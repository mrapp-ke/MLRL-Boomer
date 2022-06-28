#include "seco/learner_seco.hpp"
#include "seco/heuristics/heuristic_accuracy.hpp"
#include "seco/heuristics/heuristic_laplace.hpp"
#include "seco/heuristics/heuristic_recall.hpp"
#include "seco/heuristics/heuristic_wra.hpp"
#include "seco/rule_evaluation/head_type_partial.hpp"
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/pruning/pruning_irep.hpp"


namespace seco {

    MultiLabelSeCoRuleLearner::Config::Config() {
        this->useCoverageStoppingCriterion();
        this->useSizeStoppingCriterion();
        this->useLabelWiseStratifiedInstanceSampling();
        this->useIrepPruning();
        this->useFMeasureHeuristic();
        this->useAccuracyPruningHeuristic();
        this->usePeakLiftFunction();
        this->useParallelRuleRefinement();
        this->useParallelPrediction();
    }

    ICoverageStoppingCriterionConfig& MultiLabelSeCoRuleLearner::Config::useCoverageStoppingCriterion() {
        std::unique_ptr<CoverageStoppingCriterionConfig> ptr = std::make_unique<CoverageStoppingCriterionConfig>();
        ICoverageStoppingCriterionConfig& ref = *ptr;
        coverageStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    void MultiLabelSeCoRuleLearner::Config::usePartialHeads() {
        headConfigPtr_ = std::make_unique<PartialHeadConfig>(heuristicConfigPtr_, pruningHeuristicConfigPtr_,
                                                             liftFunctionConfigPtr_);
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

    void MultiLabelSeCoRuleLearner::Config::useRecallPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<RecallConfig>();
    }

    void MultiLabelSeCoRuleLearner::Config::useWraPruningHeuristic() {
        pruningHeuristicConfigPtr_ = std::make_unique<WraConfig>();
    }

    IBeamSearchTopDownRuleInductionConfig& MultiLabelSeCoRuleLearner::Config::useBeamSearchTopDownRuleInduction() {
        std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
            std::make_unique<BeamSearchTopDownRuleInductionConfig>(parallelRuleRefinementConfigPtr_);
        IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
        ruleInductionConfigPtr_ = std::move(ptr);
        ref.setRecalculatePredictions(false);
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

    MultiLabelSeCoRuleLearner::MultiLabelSeCoRuleLearner(std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr)
        : AbstractSeCoRuleLearner(*configPtr), configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> createMultiLabelSeCoRuleLearnerConfig() {
        return std::make_unique<MultiLabelSeCoRuleLearner::Config>();
    }

    std::unique_ptr<IMultiLabelSeCoRuleLearner> createMultiLabelSeCoRuleLearner(
            std::unique_ptr<IMultiLabelSeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<MultiLabelSeCoRuleLearner>(std::move(configPtr));
    }

}
