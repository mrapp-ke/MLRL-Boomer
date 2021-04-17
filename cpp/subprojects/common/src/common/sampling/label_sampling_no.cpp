#include "common/sampling/label_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


NoLabelSubSampling::NoLabelSubSampling(uint32 numLabels)
    : numLabels_(numLabels) {

}

std::unique_ptr<IIndexVector> NoLabelSubSampling::subSample(RNG& rng) const {
    return std::make_unique<FullIndexVector>(numLabels_);
}

std::unique_ptr<ILabelSubSampling> NoLabelSubSamplingFactory::create(uint32 numLabels) const {
    return std::make_unique<NoLabelSubSampling>(numLabels);
}
