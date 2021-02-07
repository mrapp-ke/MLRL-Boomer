#include "common/sampling/instance_sampling_random.hpp"
#include "common/sampling/weight_sampling.hpp"


RandomInstanceSubsetSelection::RandomInstanceSubsetSelection(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(uint32 numExamples, RNG& rng) const {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement(numExamples, numSamples, rng);
}
