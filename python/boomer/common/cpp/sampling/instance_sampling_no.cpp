#include "instance_sampling_no.h"
#include "weight_vector_equal.h"


std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(uint32 numExamples, RNG& rng) const {
    return std::make_unique<EqualWeightVector>(numExamples);
}
