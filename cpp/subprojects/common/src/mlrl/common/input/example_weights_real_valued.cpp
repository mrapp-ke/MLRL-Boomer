#include "mlrl/common/input/example_weights_real_valued.hpp"

#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/partition.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"

/**
 * Provides access to the weights of individual training examples in cases where the examples have real-valued weights.
 */
class RealValuedExampleWeights final : public IRealValuedExampleWeights {
    private:

        DenseWeightVector<float32> weightVector_;

    public:

        /**
         * @param numExamples The total number of available examples
         */
        RealValuedExampleWeights(uint32 numExamples) : weightVector_(DenseWeightVector<float32>(numExamples)) {}

        void setWeight(uint32 index, float32 weight) override {
            weightVector_[index] = weight;
        }

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

std::unique_ptr<IRealValuedExampleWeights> createRealValuedExampleWeights(uint32 numExamples) {
    return std::make_unique<RealValuedExampleWeights>(numExamples);
}
