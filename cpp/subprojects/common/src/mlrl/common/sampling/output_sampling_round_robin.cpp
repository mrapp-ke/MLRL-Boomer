#include "mlrl/common/sampling/output_sampling_round_robin.hpp"

#include "mlrl/common/indices/index_vector_partial.hpp"

/**
 * Allows to select one output at a time in a round-robin fashion.
 */
class RoundRobinOutputSampling final : public IOutputSampling {
    private:

        const uint32 numOutputs_;

        PartialIndexVector indexVector_;

        uint32 nextIndex_;

    public:

        /**
         * @param numOutputs The total number of available outputs
         */
        RoundRobinOutputSampling(uint32 numOutputs) : numOutputs_(numOutputs), indexVector_(1), nextIndex_(0) {}

        const IIndexVector& sample(RNG& rng) override {
            indexVector_.begin()[0] = nextIndex_;
            nextIndex_++;

            if (nextIndex_ >= numOutputs_) {
                nextIndex_ = 0;
            }

            return indexVector_;
        }
};

/**
 * Allows to create objects of type `IOutputSampling` that selects one output at a time in a round-robin fashion.
 */
class RoundRobinOutputSamplingFactory final : public IOutputSamplingFactory {
    private:

        const uint32 numOutputs_;

    public:

        /**
         * @param numOutputs The total number of available outputs
         */
        RoundRobinOutputSamplingFactory(uint32 numOutputs) : numOutputs_(numOutputs) {}

        std::unique_ptr<IOutputSampling> create() const override {
            return std::make_unique<RoundRobinOutputSampling>(numOutputs_);
        }
};

std::unique_ptr<IOutputSamplingFactory> RoundRobinOutputSamplingConfig::createOutputSamplingFactory(
  const IOutputMatrix& outputMatrix) const {
    return std::make_unique<RoundRobinOutputSamplingFactory>(outputMatrix.getNumOutputs());
}
