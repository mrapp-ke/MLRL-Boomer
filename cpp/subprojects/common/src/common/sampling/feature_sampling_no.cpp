#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


NoFeatureSubSampling::NoFeatureSubSampling(uint32 numFeatures)
    : numFeatures_(numFeatures) {

}

std::unique_ptr<IIndexVector> NoFeatureSubSampling::subSample(RNG& rng) const {
    return std::make_unique<FullIndexVector>(numFeatures_);
}

std::unique_ptr<IFeatureSubSampling> NoFeatureSubSamplingFactory::create(uint32 numFeatures) const {
    return std::make_unique<NoFeatureSubSampling>(numFeatures);
}
