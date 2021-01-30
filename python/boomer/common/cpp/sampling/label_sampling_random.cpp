#include "label_sampling_random.h"
#include "index_sampling.h"
#include "../indices/index_iterator.h"

RandomLabelSubsetSelection::RandomLabelSubsetSelection(uint32 numSamples)
    : numSamples_(numSamples) {

}

std::unique_ptr<IIndexVector> RandomLabelSubsetSelection::subSample(uint32 numLabels, RNG& rng) const {
    return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numLabels), numLabels, numSamples_, rng);
}
