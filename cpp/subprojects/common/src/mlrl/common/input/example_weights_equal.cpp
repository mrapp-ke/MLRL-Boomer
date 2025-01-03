#include "mlrl/common/input/example_weights_equal.hpp"

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
};

std::unique_ptr<IEqualExampleWeights> createEqualExampleWeights(uint32 numExamples) {
    return std::make_unique<EqualExampleWeights>(numExamples);
}
