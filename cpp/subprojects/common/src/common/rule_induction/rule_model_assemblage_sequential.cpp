#include "common/rule_induction/rule_model_assemblage_sequential.hpp"


static inline IStoppingCriterion::Result testStoppingCriteria(
        std::forward_list<std::shared_ptr<IStoppingCriterion>>& stoppingCriteria, const IPartition& partition,
        const IStatistics& statistics, uint32 numRules) {
    IStoppingCriterion::Result result;
    result.action = IStoppingCriterion::Action::CONTINUE;

    for (auto it = stoppingCriteria.begin(); it != stoppingCriteria.end(); it++) {
        std::shared_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;
        IStoppingCriterion::Result stoppingCriterionResult = stoppingCriterionPtr->test(partition, statistics,
                                                                                        numRules);
        IStoppingCriterion::Action action = stoppingCriterionResult.action;

        switch (action) {
            case IStoppingCriterion::Action::FORCE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                return result;
            }
            case IStoppingCriterion::Action::STORE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                break;
            }
            default: {
                break;
            }
        }
    }

    return result;
}

SequentialRuleModelAssemblage::SequentialRuleModelAssemblage(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr, std::shared_ptr<IPruning> pruningPtr,
        std::shared_ptr<IPostProcessor> postProcessorPtr,
        std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
      ruleInductionPtr_(ruleInductionPtr), defaultRuleHeadRefinementFactoryPtr_(defaultRuleHeadRefinementFactoryPtr),
      headRefinementFactoryPtr_(headRefinementFactoryPtr), labelSamplingFactoryPtr_(labelSamplingFactoryPtr),
      instanceSamplingFactoryPtr_(instanceSamplingFactoryPtr), featureSamplingFactoryPtr_(featureSamplingFactoryPtr),
      partitionSamplingFactoryPtr_(partitionSamplingFactoryPtr), pruningPtr_(pruningPtr),
      postProcessorPtr_(postProcessorPtr), stoppingCriteriaPtr_(std::move(stoppingCriteriaPtr)) {

}

std::unique_ptr<RuleModel> SequentialRuleModelAssemblage::induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                                      const IFeatureMatrix& featureMatrix,
                                                                      const ILabelMatrix& labelMatrix,
                                                                      uint32 randomState, IModelBuilder& modelBuilder) {
    // Induce default rule...
    const IHeadRefinementFactory* defaultRuleHeadRefinementFactory = defaultRuleHeadRefinementFactoryPtr_.get();
    uint32 numRules = defaultRuleHeadRefinementFactory != nullptr ? 1 : 0;
    uint32 numUsedRules = 0;
    std::unique_ptr<IStatisticsProvider> statisticsProviderPtr = labelMatrix.createStatisticsProvider(
        *statisticsProviderFactoryPtr_);
    ruleInductionPtr_->induceDefaultRule(*statisticsProviderPtr, defaultRuleHeadRefinementFactory, modelBuilder);

    // Induce the remaining rules...
    std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrix, nominalFeatureMask,
                                                                               *statisticsProviderPtr,
                                                                               *headRefinementFactoryPtr_);
    uint32 numFeatures = thresholdsPtr->getNumFeatures();
    uint32 numLabels = thresholdsPtr->getNumLabels();
    std::unique_ptr<IPartitionSampling> partitionSamplingPtr = labelMatrix.createPartitionSampling(
        *partitionSamplingFactoryPtr_);
    RNG rng(randomState);
    IPartition& partition = partitionSamplingPtr->partition(rng);
    std::unique_ptr<IInstanceSampling> instanceSamplingPtr = partition.createInstanceSampling(
        *instanceSamplingFactoryPtr_, labelMatrix, statisticsProviderPtr->get());
    std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr_->create(numFeatures);
    std::unique_ptr<ILabelSampling> labelSamplingPtr = labelSamplingFactoryPtr_->create(numLabels);
    IStoppingCriterion::Result stoppingCriterionResult;

    while (stoppingCriterionResult = testStoppingCriteria(*stoppingCriteriaPtr_, partition,
                                                          statisticsProviderPtr->get(), numRules),
           stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
        if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
            numUsedRules = stoppingCriterionResult.numRules;
        }

        const IWeightVector& weights = instanceSamplingPtr->sample(rng);
        const IIndexVector& labelIndices = labelSamplingPtr->sample(rng);
        bool success = ruleInductionPtr_->induceRule(*thresholdsPtr, labelIndices, weights, partition,
                                                     *featureSamplingPtr, *pruningPtr_, *postProcessorPtr_, rng,
                                                     modelBuilder);

        if (success) {
            numRules++;
        } else {
            break;
        }
    }

    // Build and return the final model...
    return modelBuilder.build(numUsedRules);
}
