#include "common/sampling/partition_sampling_no.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * An implementation of the class `IPartitionSampling` that does not split the training examples, but includes all of
 * them in the training set.
 */
class NoPartitionSampling final : public IPartitionSampling {

    private:

        uint32 numExamples_;

    public:

        NoPartitionSampling(uint32 numExamples)
            : numExamples_(numExamples) {

        }

        std::unique_ptr<IPartition> createPartition(RNG& rng) const override {
            return std::make_unique<SinglePartition>(numExamples_);
        }

};

std::unique_ptr<IPartitionSampling> NoPartitionSamplingFactory::create(uint32 numExamples) const {
    return std::make_unique<NoPartitionSampling>(numExamples);
}
