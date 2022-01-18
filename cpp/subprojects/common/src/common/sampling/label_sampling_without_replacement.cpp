#include "common/sampling/label_sampling_without_replacement.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/util/validation.hpp"
#include "index_sampling.hpp"


/**
 * Allows to select a subset of the available labels without replacement.
 */
class LabelSamplingWithoutReplacement final : public ILabelSampling {

    private:

        uint32 numLabels_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numLabels     The total number of available labels
         * @param numSamples    The number of labels to be included in the sample
         */
        LabelSamplingWithoutReplacement(uint32 numLabels, uint32 numSamples)
            : numLabels_(numLabels), indexVector_(PartialIndexVector(numSamples)) {

        }

        const IIndexVector& sample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_, IndexIterator(numLabels_), numLabels_, rng);
            return indexVector_;
        }

};

/**
 * Allows to create objects of type `ILabelSampling` that select a random subset of the available features without
 * replacement.
 */
class LabelSamplingWithoutReplacementFactory final : public ILabelSamplingFactory {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param numSamples The number of labels to be included in the sample. Must be at least 1
         */
        LabelSamplingWithoutReplacementFactory(uint32 numSamples)
            : numSamples_(numSamples) {

        }

        std::unique_ptr<ILabelSampling> create(uint32 numLabels) const override {
            uint32 numSamples = numSamples_ > numLabels ? numLabels : numSamples_;
            return std::make_unique<LabelSamplingWithoutReplacement>(numLabels, numSamples);
        }

};

LabelSamplingWithoutReplacementConfig::LabelSamplingWithoutReplacementConfig()
    : numSamples_(1) {

}

uint32 LabelSamplingWithoutReplacementConfig::getNumSamples() const {
    return numSamples_;
}

ILabelSamplingWithoutReplacementConfig& LabelSamplingWithoutReplacementConfig::setNumSamples(uint32 numSamples) {
    assertGreaterOrEqual<uint32>("numSamples", numSamples, 1);
    numSamples_ = numSamples;
    return *this;
}

std::unique_ptr<ILabelSamplingFactory> LabelSamplingWithoutReplacementConfig::create() const {
    return std::make_unique<LabelSamplingWithoutReplacementFactory>(numSamples_);
}
