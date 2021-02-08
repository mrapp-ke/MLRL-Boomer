#include "common/sampling/instance_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "weight_sampling.hpp"


RandomInstanceSubsetSelection::RandomInstanceSubsetSelection(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(uint32 numExamples, RNG& rng) const {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement<IndexIterator>(IndexIterator(numExamples), numExamples, numSamples,
                                                          numExamples, rng);
}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(std::unique_ptr<SinglePartition> partitionPtr,
                                                                        RNG& rng) const {
    // TODO
}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(std::unique_ptr<BiPartition> partitionPtr,
                                                                        RNG& rng) const {
    // TODO
}
