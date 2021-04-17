#include "common/sampling/label_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"


/**
 * Allows to select a subset of the available labels without replacement.
 */
class RandomLabelSubsetSelection final : public ILabelSubSampling {

    private:

        uint32 numLabels_;

        uint32 numSamples_;

    public:

        /**
         * @param numLabels     The total number of available labels
         * @param numSamples    The number of labels to be included in the sample
         */
        RandomLabelSubsetSelection(uint32 numLabels, uint32 numSamples)
            : numLabels_(numLabels), numSamples_(numSamples) {

        }

        std::unique_ptr<IIndexVector> subSample(RNG& rng) const override {
            return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numLabels_), numLabels_, numSamples_,
                                                                  rng);
        }

};

std::unique_ptr<ILabelSubSampling> RandomLabelSubsetSelectionFactory::create(uint32 numLabels) const {
    return std::make_unique<RandomLabelSubsetSelection>(numLabels, numSamples_);
}
