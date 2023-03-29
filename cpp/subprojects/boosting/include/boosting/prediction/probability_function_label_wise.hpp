/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to transform regression scores that are predicted for individual
 * labels into probabilities.
 */
class ILabelWiseProbabilityFunction {
    public:

        virtual ~ILabelWiseProbabilityFunction() {};

        /**
         * Transforms the regression score that is predicted for an individual label into a probability.
         *
         * @param predictedScore    The regression score that is predicted for a label
         * @return                  The probability
         */
        virtual float64 transform(float64 predictedScore) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `ILabelWiseProbabilityFunction`.
 */
class ILabelWiseProbabilityFunctionFactory {
    public:

        virtual ~ILabelWiseProbabilityFunctionFactory() {};

        /**
         * Creates and returns a new object of the type `ILabelWiseProbabilityFunction`.
         *
         * @return An unique pointer to an object of type `ILabelWiseProbabilityFunction` that has been created
         */
        virtual std::unique_ptr<ILabelWiseProbabilityFunction> create() const = 0;
};
