#include "mlrl/common/sampling/output_sampling_no.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"

/**
 * An implementation of the class `IOutputSampling` that does not perform any sampling, but includes all outputs.
 */
class NoOutputSampling final : public IOutputSampling {
    private:

        const CompleteIndexVector indexVector_;

    public:

        /**
         * @param numOutputs The total number of available outputs
         */
        NoOutputSampling(uint32 numOutputs) : indexVector_(numOutputs) {}

        const IIndexVector& sample(RNG& rng) override {
            return indexVector_;
        }
};

/**
 * Allows to create objects of the class `IOutputSampling` that do not perform any sampling, but includes all outputs.
 */
class NoOutputSamplingFactory final : public IOutputSamplingFactory {
    private:

        const uint32 numOutputs_;

    public:

        /**
         * @param numOutputs The total number of available outputs
         */
        NoOutputSamplingFactory(uint32 numOutputs) : numOutputs_(numOutputs) {}

        std::unique_ptr<IOutputSampling> create() const override {
            return std::make_unique<NoOutputSampling>(numOutputs_);
        }
};

std::unique_ptr<IOutputSamplingFactory> NoOutputSamplingConfig::createOutputSamplingFactory(
  const IOutputMatrix& outputMatrix) const {
    return std::make_unique<NoOutputSamplingFactory>(outputMatrix.getNumOutputs());
}
