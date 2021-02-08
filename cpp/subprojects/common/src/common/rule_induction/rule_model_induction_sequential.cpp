#include "common/rule_induction/rule_model_induction_sequential.hpp"


static inline bool shouldContinue(std::forward_list<std::shared_ptr<IStoppingCriterion>>& stoppingCriteria,
                                  const IStatistics& statistics, uint32 numRules) {
    for (auto it = stoppingCriteria.begin(); it != stoppingCriteria.end(); it++) {
        std::shared_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;

        if (!stoppingCriterionPtr->shouldContinue(statistics, numRules)) {
            return false;
        }
    }

    return true;
}

SequentialRuleModelInduction::SequentialRuleModelInduction(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
        std::shared_ptr<ILabelSubSampling> labelSubSamplingPtr,
        std::shared_ptr<IInstanceSubSampling> instanceSubSamplingPtr,
        std::shared_ptr<IFeatureSubSampling> featureSubSamplingPtr,
        std::shared_ptr<IPartitionSampling> partitionSamplingPtr, std::shared_ptr<IPruning> pruningPtr,
        std::shared_ptr<IPostProcessor> postProcessorPtr, uint32 minCoverage, intp maxConditions,
        intp maxHeadRefinements,
        std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
      ruleInductionPtr_(ruleInductionPtr), defaultRuleHeadRefinementFactoryPtr_(defaultRuleHeadRefinementFactoryPtr),
      headRefinementFactoryPtr_(headRefinementFactoryPtr), labelSubSamplingPtr_(labelSubSamplingPtr),
      instanceSubSamplingPtr_(instanceSubSamplingPtr), featureSubSamplingPtr_(featureSubSamplingPtr),
      partitionSamplingPtr_(partitionSamplingPtr), pruningPtr_(pruningPtr), postProcessorPtr_(postProcessorPtr),
      minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
      stoppingCriteriaPtr_(std::move(stoppingCriteriaPtr)) {

}

std::unique_ptr<RuleModel> SequentialRuleModelInduction::induceRules(
        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr, std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
        std::shared_ptr<ILabelMatrix> labelMatrixPtr, RNG& rng, IModelBuilder& modelBuilder) {
    // Induce default rule...
    const IHeadRefinementFactory* defaultRuleHeadRefinementFactory = defaultRuleHeadRefinementFactoryPtr_.get();
    uint32 numRules = defaultRuleHeadRefinementFactory != nullptr ? 1 : 0;
     std::shared_ptr<IRandomAccessLabelMatrix> randomAccessLabelMatrixPtr =
        std::dynamic_pointer_cast<IRandomAccessLabelMatrix, ILabelMatrix>(labelMatrixPtr);
    std::shared_ptr<IStatisticsProvider> statisticsProviderPtr = statisticsProviderFactoryPtr_->create(
        randomAccessLabelMatrixPtr);
    ruleInductionPtr_->induceDefaultRule(*statisticsProviderPtr, defaultRuleHeadRefinementFactory, modelBuilder);

    // Induce the remaining rules...
    std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrixPtr, nominalFeatureMaskPtr,
                                                                               statisticsProviderPtr,
                                                                               headRefinementFactoryPtr_);
    uint32 numExamples = thresholdsPtr->getNumExamples();
    uint32 numLabels = thresholdsPtr->getNumLabels();
    std::unique_ptr<IPartition> partitionPtr = partitionSamplingPtr_->partition(numExamples, rng);

    while (shouldContinue(*stoppingCriteriaPtr_, statisticsProviderPtr->get(), numRules)) {
        std::unique_ptr<IWeightVector> weightsPtr = partitionPtr->subSample(*instanceSubSamplingPtr_, rng);
        std::unique_ptr<IIndexVector> labelIndicesPtr = labelSubSamplingPtr_->subSample(numLabels, rng);
        bool success = ruleInductionPtr_->induceRule(*thresholdsPtr, *labelIndicesPtr, *weightsPtr, *partitionPtr,
                                                     *featureSubSamplingPtr_, *pruningPtr_, *postProcessorPtr_,
                                                     minCoverage_, maxConditions_, maxHeadRefinements_, rng,
                                                     modelBuilder);

        if (success) {
            numRules++;
        } else {
            break;
        }
    }

    // Build and return the final model...
    return modelBuilder.build();
}
