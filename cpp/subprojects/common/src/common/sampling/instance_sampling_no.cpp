#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"


std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(uint32 numExamples, RNG& rng) const {
    return std::make_unique<EqualWeightVector>(numExamples);
}
