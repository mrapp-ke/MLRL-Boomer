#include "rule_model_induction_sequential.h"


SequentialRuleModelInduction::SequentialRuleModelInduction(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
        std::shared_ptr<ILabelSubSampling> labelSubSamplingPtr,
        std::shared_ptr<IInstanceSubSampling> instanceSubSamplingPtr,
        std::shared_ptr<IFeatureSubSampling> featureSubSamplingPtr, std::shared_ptr<IPruning> pruningPtr,
        std::shared_ptr<IPostProcessor> postProcessorPtr, uint32 minCoverage, intp maxConditions,
        intp maxHeadRefinements, uint32 numThreads, uint32 randomState,
        std::unique_ptr<std::forward_list<IStoppingCriterion>> stoppingCriteriaPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), ruleInductionPtr_(ruleInductionPtr),
      defaultRuleHeadRefinementFactoryPtr_(defaultRuleHeadRefinementFactoryPtr),
      headRefinementFactoryPtr_(headRefinementFactoryPtr), labelSubSamplingPtr_(labelSubSamplingPtr),
      instanceSubSamplingPtr_(instanceSubSamplingPtr), featureSubSamplingPtr_(featureSubSamplingPtr),
      pruningPtr_(pruningPtr), postProcessorPtr_(postProcessorPtr), minCoverage_(minCoverage),
      maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements), numThreads_(numThreads),
      stoppingCriteriaPtr_(std::move(stoppingCriteriaPtr)), rng_(RNG(randomState)) {

}

std::unique_ptr<RuleModel> SequentialRuleModelInduction::induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                                     const IFeatureMatrix& featureMatrix,
                                                                     const ILabelMatrix& labelMatrix,
                                                                     IModelBuilder& modelBuilder) const {
    // TODO
    return std::make_unique<RuleModel>();
}
