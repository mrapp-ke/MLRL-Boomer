#include "common/sampling/partition_sampling_bi.hpp"
#include "common/sampling/partition_bi.hpp"
#include "index_sampling.hpp"


BiPartitionSampling::BiPartitionSampling(uint32 numExamples, float32 holdoutSetSize)
    : numHoldout_((uint32) (holdoutSetSize * numExamples)), numTraining_(numExamples - numHoldout_) {

}

std::unique_ptr<IPartition> BiPartitionSampling::partition(RNG& rng) const {
    std::unique_ptr<BiPartition> partitionPtr = std::make_unique<BiPartition>(numTraining_, numHoldout_);
    BiPartition::iterator trainingIterator = partitionPtr->first_begin();
    BiPartition::iterator holdoutIterator = partitionPtr->second_begin();

    for (uint32 i = 0; i < numTraining_; i++) {
        trainingIterator[i] = i;
    }

    for (uint32 i = 0; i < numHoldout_; i++) {
        holdoutIterator[i] = numTraining_ + i;
    }

    randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator, numTraining_,
                                                                    partitionPtr->getNumElements(), rng);
    return partitionPtr;
}

BiPartitionSamplingFactory::BiPartitionSamplingFactory(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {

}

std::unique_ptr<IPartitionSampling> BiPartitionSamplingFactory::create(uint32 numExamples) const {
    return std::make_unique<BiPartitionSampling>(numExamples, holdoutSetSize_);
}
