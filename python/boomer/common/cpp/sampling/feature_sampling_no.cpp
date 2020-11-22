#include "feature_sampling_no.h"
#include "../indices/index_vector_full.h"


std::unique_ptr<IIndexVector> NoFeatureSubSampling::subSample(uint32 numFeatures, RNG& rng) const {
    return std::make_unique<FullIndexVector>(numFeatures);
}
