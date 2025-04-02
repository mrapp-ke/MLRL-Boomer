/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/input/example_weights.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to the weights of individual training examples in cases
 * where the examples have real-valued weights.
 */
class MLRLCOMMON_API IRealValuedExampleWeights : public IExampleWeights {
    public:

        virtual ~IRealValuedExampleWeights() override {}

        /**
         * Sets the weight of the training example at a specific index.
         *
         * @param index     The index of the training example
         * @param weight    The weight to be set
         */
        virtual void setWeight(uint32 index, float32 weight) = 0;
};

/**
 * Creates and returns a new object of type `IRealValuedExampleWeights` in cases where the examples have real-valued
 * weights.
 *
 * @param numExamples   The total number of available examples
 * @return              An unique pointer to an object of type `IRealValuedExampleWeights` that have been created
 */
MLRLCOMMON_API std::unique_ptr<IRealValuedExampleWeights> createRealValuedExampleWeights(uint32 numExamples);
