#include "mlrl/common/input/example_weights_equal.hpp"

#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"

/**
 * Provides access to the weights of individual training examples in cases where all examples have equal weights.
 */
class EqualExampleWeights final : public IEqualExampleWeights {
    private:

        EqualWeightVector weightVector_;

    public:

        /**
         * @param numExamples The total number of available examples
         */
        EqualExampleWeights(uint32 numExamples) : weightVector_(EqualWeightVector(numExamples)) {}

        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
          IStatistics& statistics, IPartition& partition) const override {
            return partition.createInstanceSampling(factory, labelMatrix, statistics, weightVector_);
        }

        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
          IStatistics& statistics, IPartition& partition) const override {
            return partition.createInstanceSampling(factory, regressionMatrix, statistics, weightVector_);
        }
};

std::unique_ptr<IEqualExampleWeights> createEqualExampleWeights(uint32 numExamples) {
    return std::make_unique<EqualExampleWeights>(numExamples);
}
