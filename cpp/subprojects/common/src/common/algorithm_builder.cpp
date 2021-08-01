#include "common/algorithm_builder.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"


AlgorithmBuilder::AlgorithmBuilder(std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                                   std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                                   std::shared_ptr<IRuleInduction> ruleInductionPtr,
                                   std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                                   std::shared_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
      ruleInductionPtr_(ruleInductionPtr), regularRuleHeadRefinementFactoryPtr_(headRefinementFactoryPtr),
      ruleModelAssemblageFactoryPtr_(ruleModelAssemblageFactoryPtr),
      labelSamplingFactoryPtr_(std::make_shared<NoLabelSamplingFactory>()),
      instanceSamplingFactoryPtr_(std::make_shared<NoInstanceSamplingFactory>()),
      featureSamplingFactoryPtr_(std::make_shared<NoFeatureSamplingFactory>()),
      partitionSamplingFactoryPtr_(std::make_shared<NoPartitionSamplingFactory>()),
      pruningPtr_(std::make_shared<NoPruning>()), postProcessorPtr_(std::make_shared<NoPostProcessor>()) {

}

AlgorithmBuilder& AlgorithmBuilder::setDefaultRuleHeadRefinementFactory(
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) {
    defaultRuleHeadRefinementFactoryPtr_ = headRefinementFactoryPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setLabelSamplingFactory(
        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr) {
    labelSamplingFactoryPtr_ = labelSamplingFactoryPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setInstanceSamplingFactory(
        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr) {
    instanceSamplingFactoryPtr_ = instanceSamplingFactoryPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setFeatureSamplingFactory(
        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr) {
    featureSamplingFactoryPtr_ = featureSamplingFactoryPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPartitionSamplingFactory(
        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr) {
    partitionSamplingFactoryPtr_ = partitionSamplingFactoryPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPruning(std::shared_ptr<IPruning> pruningPtr) {
    pruningPtr_ = pruningPtr;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPostProcessor(std::unique_ptr<IPostProcessor> postProcessorPtr) {
    postProcessorPtr_ = std::move(postProcessorPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::addStoppingCriterion(std::shared_ptr<IStoppingCriterion> stoppingCriterionPtr) {
    stoppingCriteria_.push_front(stoppingCriterionPtr);
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
