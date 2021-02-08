#include "common/sampling/partition_single.hpp"
#include "common/sampling/instance_sampling.hpp"


SinglePartition::SinglePartition(uint32 numElements)
    : numElements_(numElements) {

}

SinglePartition::const_iterator SinglePartition::cbegin() const {
    return IndexIterator(0);
}

SinglePartition::const_iterator SinglePartition::cend() const {
    return IndexIterator(numElements_);
}

uint32 SinglePartition::getNumElements() const {
    return numElements_;
}

std::unique_ptr<IWeightVector> SinglePartition::subSample(const IInstanceSubSampling& instanceSubSampling,
                                                          RNG& rng) const {
    return instanceSubSampling.subSample(*this, rng);
}
