#include "common/sampling/label_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"

RandomLabelSubsetSelection::RandomLabelSubsetSelection(uint32 numLabels, uint32 numSamples)
    : numLabels_(numLabels), numSamples_(numSamples) {

}

std::unique_ptr<IIndexVector> RandomLabelSubsetSelection::subSample(RNG& rng) const {
    return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numLabels_), numLabels_, numSamples_, rng);
}

std::unique_ptr<ILabelSubSampling> RandomLabelSubsetSelectionFactory::create(uint32 numLabels) const {
    return std::make_unique<RandomLabelSubsetSelection>(numLabels, numSamples_);
}
