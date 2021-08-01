#include "common/algorithm_builder.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"


AlgorithmBuilder::AlgorithmBuilder(std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                                   std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                                   std::unique_ptr<IRuleInduction> ruleInductionPtr,
                                   std::unique_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                                   std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr)
    : statisticsProviderFactoryPtr_(std::move(statisticsProviderFactoryPtr)),
      thresholdsFactoryPtr_(std::move(thresholdsFactoryPtr)), ruleInductionPtr_(std::move(ruleInductionPtr)),
      regularRuleHeadRefinementFactoryPtr_(std::move(headRefinementFactoryPtr)),
      ruleModelAssemblageFactoryPtr_(std::move(ruleModelAssemblageFactoryPtr)),
      labelSamplingFactoryPtr_(std::make_shared<NoLabelSamplingFactory>()),
      instanceSamplingFactoryPtr_(std::make_shared<NoInstanceSamplingFactory>()),
      featureSamplingFactoryPtr_(std::make_shared<NoFeatureSamplingFactory>()),
      partitionSamplingFactoryPtr_(std::make_shared<NoPartitionSamplingFactory>()),
      pruningPtr_(std::make_shared<NoPruning>()), postProcessorPtr_(std::make_shared<NoPostProcessor>()) {

}

AlgorithmBuilder& AlgorithmBuilder::setDefaultRuleHeadRefinementFactory(
        std::unique_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) {
    defaultRuleHeadRefinementFactoryPtr_ = std::move(headRefinementFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setLabelSamplingFactory(
        std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr) {
    labelSamplingFactoryPtr_ = std::move(labelSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setInstanceSamplingFactory(
        std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr) {
    instanceSamplingFactoryPtr_ = std::move(instanceSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setFeatureSamplingFactory(
        std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr) {
    featureSamplingFactoryPtr_ = std::move(featureSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPartitionSamplingFactory(
        std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr) {
    partitionSamplingFactoryPtr_ = std::move(partitionSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPruning(std::unique_ptr<IPruning> pruningPtr) {
    pruningPtr_ = std::move(pruningPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPostProcessor(std::unique_ptr<IPostProcessor> postProcessorPtr) {
    postProcessorPtr_ = std::move(postProcessorPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::addStoppingCriterion(std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr) {
    stoppingCriteria_.push_front(std::move(stoppingCriterionPtr));
    return *this;
}

std::unique_ptr<IRuleModelAssemblage> AlgorithmBuilder::build() const {
    return ruleModelAssemblageFactoryPtr_->create(statisticsProviderFactoryPtr_, thresholdsFactoryPtr_,
                                                  ruleInductionPtr_, defaultRuleHeadRefinementFactoryPtr_,
                                                  regularRuleHeadRefinementFactoryPtr_, labelSamplingFactoryPtr_,
                                                  instanceSamplingFactoryPtr_, featureSamplingFactoryPtr_,
                                                  partitionSamplingFactoryPtr_, pruningPtr_, postProcessorPtr_,
                                                  stoppingCriteria_);
}
