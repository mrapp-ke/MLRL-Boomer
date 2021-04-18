#include "common/sampling/label_sampling_random.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"


/**
 * Allows to select a subset of the available labels without replacement.
 */
class RandomLabelSubsetSelection final : public ILabelSubSampling {

    private:

        uint32 numLabels_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numLabels     The total number of available labels
         * @param numSamples    The number of labels to be included in the sample
         */
        RandomLabelSubsetSelection(uint32 numLabels, uint32 numSamples)
            : numLabels_(numLabels), indexVector_(PartialIndexVector(numSamples)) {

        }

        const IIndexVector& subSample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_.begin(), indexVector_.getNumElements(),
                                                           IndexIterator(numLabels_), numLabels_, rng);
            return indexVector_;
        }

};

RandomLabelSubsetSelectionFactory::RandomLabelSubsetSelectionFactory(uint32 numSamples)
    : numSamples_(numSamples) {

}

std::unique_ptr<ILabelSubSampling> RandomLabelSubsetSelectionFactory::create(uint32 numLabels) const {
    return std::make_unique<RandomLabelSubsetSelection>(numLabels, numSamples_);
}
