#include "label_sampling_no.hpp"
#include "../indices/index_vector_full.hpp"


std::unique_ptr<IIndexVector> NoLabelSubSampling::subSample(uint32 numLabels, RNG& rng) const {
    return std::make_unique<FullIndexVector>(numLabels);
}
