#include "mlrl/common/sampling/partition_sampling_no.hpp"

#include "mlrl/common/sampling/partition_single.hpp"

/**
 * An implementation of the class `IPartitionSampling` that does not split the training examples, but includes all of
 * them in the training set.
 */
class NoPartitionSampling final : public IPartitionSampling {
    private:

        SinglePartition partition_;

    public:

        /**
         * @param numExamples The total number of available training examples
         */
        NoPartitionSampling(uint32 numExamples) : partition_(numExamples) {}

        IPartition& partition() override {
            return partition_;
        }
};

template<typename OutputMatrix>
static inline std::unique_ptr<IPartitionSampling> createNoPartitionSampling(const OutputMatrix& outputMatrix) {
    return std::make_unique<NoPartitionSampling>(outputMatrix.numRows);
}

/**
 * Allows to create objects of the type `IPartitionSampling` that do not split the training examples, but include all of
 * them in the training set.
 */
class NoPartitionSamplingFactory final : public IClassificationPartitionSamplingFactory,
                                         public IRegressionPartitionSamplingFactory {
    public:

        std::unique_ptr<IPartitionSampling> create(const CContiguousView<const uint8>& labelMatrix) const override {
            return createNoPartitionSampling(labelMatrix);
        }

        std::unique_ptr<IPartitionSampling> create(const BinaryCsrView& labelMatrix) const override {
            return createNoPartitionSampling(labelMatrix);
        }

        std::unique_ptr<IPartitionSampling> create(
          const CContiguousView<const float32>& regressionMatrix) const override {
            return createNoPartitionSampling(regressionMatrix);
        }

        std::unique_ptr<IPartitionSampling> create(const CsrView<const float32>& regressionMatrix) const override {
            return createNoPartitionSampling(regressionMatrix);
        }
};

std::unique_ptr<IClassificationPartitionSamplingFactory>
  NoPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
    return std::make_unique<NoPartitionSamplingFactory>();
}

std::unique_ptr<IRegressionPartitionSamplingFactory>
  NoPartitionSamplingConfig::createRegressionPartitionSamplingFactory() const {
    return std::make_unique<NoPartitionSamplingFactory>();
}
