#include "common/sampling/feature_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"
#include <cmath>


RandomFeatureSubsetSelection::RandomFeatureSubsetSelection(uint32 numFeatures, float32 sampleSize)
    : numFeatures_(numFeatures),
      numSamples_((uint32) (sampleSize > 0 ? sampleSize * numFeatures : log2(numFeatures - 1) + 1)) {

}

std::unique_ptr<IIndexVector> RandomFeatureSubsetSelection::subSample(RNG& rng) const {
    return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numFeatures_), numFeatures_, numSamples_, rng);
}

RandomFeatureSubsetSelectionFactory::RandomFeatureSubsetSelectionFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IFeatureSubSampling> RandomFeatureSubsetSelectionFactory::create(uint32 numFeatures) const {
    return std::make_unique<RandomFeatureSubsetSelection>(numFeatures, sampleSize_);
}
