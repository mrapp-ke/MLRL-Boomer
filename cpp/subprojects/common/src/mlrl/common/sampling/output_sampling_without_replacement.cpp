#include "mlrl/common/sampling/output_sampling_without_replacement.hpp"

#include "index_sampling.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Allows to select a subset of the available outputs without replacement.
 */
class OutputSamplingWithoutReplacement final : public IOutputSampling {
    private:

        const uint32 numOutputs_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numOutputs    The total number of available outputs
         * @param numSamples    The number of outputs to be included in a sample
         */
        OutputSamplingWithoutReplacement(uint32 numOutputs, uint32 numSamples)
            : numOutputs_(numOutputs), indexVector_(numSamples) {}

        const IIndexVector& sample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_.begin(), indexVector_.getNumElements(),
                                                           IndexIterator(numOutputs_), numOutputs_, rng);
            return indexVector_;
        }
};

/**
 * Allows to create objects of type `IOutputSampling` that select a subset of the available outputs without replacement.
 */
class OutputSamplingWithoutReplacementFactory final : public IOutputSamplingFactory {
    private:

        const uint32 numOutputs_;

        const uint32 numSamples_;

    public:

        /**
         * @param numOutputs    The total number of available outputs
         * @param numSamples    The number of outputs to be included in a sample. Must be at least 1
         */
        OutputSamplingWithoutReplacementFactory(uint32 numOutputs, uint32 numSamples)
            : numOutputs_(numOutputs), numSamples_(numSamples > numOutputs ? numOutputs : numSamples) {}

        std::unique_ptr<IOutputSampling> create() const override {
            return std::make_unique<OutputSamplingWithoutReplacement>(numOutputs_, numSamples_);
        }
};

OutputSamplingWithoutReplacementConfig::OutputSamplingWithoutReplacementConfig() : numSamples_(1) {}

uint32 OutputSamplingWithoutReplacementConfig::getNumSamples() const {
    return numSamples_;
}

IOutputSamplingWithoutReplacementConfig& OutputSamplingWithoutReplacementConfig::setNumSamples(uint32 numSamples) {
    util::assertGreaterOrEqual<uint32>("numSamples", numSamples, 1);
    numSamples_ = numSamples;
    return *this;
}

std::unique_ptr<IOutputSamplingFactory> OutputSamplingWithoutReplacementConfig::createOutputSamplingFactory(
  const IOutputMatrix& outputMatrix) const {
    return std::make_unique<OutputSamplingWithoutReplacementFactory>(outputMatrix.getNumOutputs(), numSamples_);
}
