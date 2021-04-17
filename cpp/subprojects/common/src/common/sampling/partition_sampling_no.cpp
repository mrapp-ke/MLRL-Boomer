#include "common/sampling/partition_sampling_no.hpp"
#include "common/sampling/partition_single.hpp"


NoPartitionSampling::NoPartitionSampling(uint32 numExamples)
    : numExamples_(numExamples) {

}

std::unique_ptr<IPartition> NoPartitionSampling::createPartition(RNG& rng) const {
    return std::make_unique<SinglePartition>(numExamples_);
}

std::unique_ptr<IPartitionSampling> NoPartitionSamplingFactory::create(uint32 numExamples) const {
    return std::make_unique<NoPartitionSampling>(numExamples);
}
