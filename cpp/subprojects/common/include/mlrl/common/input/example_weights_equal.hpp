/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/input/example_weights.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to the weights of individual training examples in cases
 * where all examples have equal weights.
 */
class MLRLCOMMON_API IEqualExampleWeights : public IExampleWeights {
    public:

        virtual ~IEqualExampleWeights() override {}
};

/**
 * Creates and returns a new object of type `IEqualExampleWeights` in cases where all examples have equal weights.
 *
 * @param numExamples   The total number of available examples
 * @return              An unique pointer to an object of type `IEqualExampleWeights` that have been created
 */
MLRLCOMMON_API std::unique_ptr<IEqualExampleWeights> createEqualExampleWeights(uint32 numExamples);
